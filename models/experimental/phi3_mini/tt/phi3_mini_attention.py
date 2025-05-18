from typing import Optional, Tuple

import torch
import ttnn
import math

from models.common.lightweightmodule import LightweightModule
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


class TtPhi3MiniAttention(LightweightModule):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        layer_idx,
        device,
        kernel_args,
        max_batch_size=32,
        transformation_mats=None,
    ):
        super().__init__()

        self.device = device
        self.num_devices = 1
        self.state_dict = state_dict
        self.mesh_device = device
        self.model_args = kernel_args
        self.model_config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_position_embeddings // 16
        self.max_batch_size = max_batch_size
        self.n_kv_heads = config.num_key_value_heads
        self.attn_output_multiplier = 1.0 / math.sqrt(self.head_dim)
        self.original_max_seq_len = config.original_max_position_embeddings
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.model_mem_configs = self.model_args.get_model_mem_configs()
        self.transformation_mats = transformation_mats
        # self.dtype = dtype # TODO: take it as arg from init

        self.wqkv = ttnn.as_tensor(
            torch.transpose(
                state_dict[f"{base_address}.qkv_proj.weight"],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_mem_configs["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                state_dict[f"{base_address}.o_proj.weight"],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_mem_configs["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
        )

        cache_k = torch.zeros(  # 12582912000 * 1 bytes values -> 24GB :(
            (
                self.max_batch_size,
                self.n_local_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.n_local_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )
        )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=ttnn.bfloat8_b,
                layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for k_or_v in [cache_k, cache_v]
        ]

        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

        # Will be filled during the initial warmup run
        self.q_mem_config = None
        self.k_mem_config = None

    def forward_decode(
        self,
        x,
        rot_mats,
        current_pos,
        attn_masks,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[Tuple[ttnn.Tensor]]]:
        """
        x: (seq_len, 1, batch, hidden_dim)
        current_pos: the length of the KV cache. Same as current token's index.
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (8)
        H : dim (3072)
        D : head_dim (96)
        P : padded_layer_past_len
        """

        ###
        # QKV Linear
        ###
        # print(f"x_11BH: {x.shape}")
        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)

        ###
        # Reshape and rotary embeddings TODO: Scaled ROPE for >4k Context length
        ###
        # print(f"fused_qkv: {xqkv_fused.shape}")
        (
            q_heads_1BQD,
            k_heads_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1BQD.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1BKD.memory_config()

        # print(f"q_heads_1BQD: {q_heads_1BQD.shape}")
        q_heads_1BQD = ttnn.matmul(
            q_heads_1BQD,
            rot_mats,
            memory_config=self.q_mem_config,
            compute_kernel_config=self.compute_kernel_attn,
            # [bsz, seqlen, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [bsz, seqlen, padd_heads, head_dim]
        )
        # print(f"k_heads_1BKD: {k_heads_1BKD.shape}")
        k_heads_1BKD = ttnn.matmul(
            k_heads_1BKD,
            rot_mats,
            memory_config=self.k_mem_config,
            compute_kernel_config=self.compute_kernel_attn,
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )

        # q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
        #     q_heads_1BQD,  # example input
        #     rot_mats[0],
        #     rot_mats[1],
        #     transformation_mats_decode,
        #     is_decode_mode=True
        # )
        # k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
        #     k_heads_1BKD, # example input
        #     rot_mats[0],
        #     rot_mats[1],
        #     transformation_mats_decode,
        #     is_decode_mode=True
        # )

        ###
        # KV update
        ###
        # print(f"v_heads_1BKD: {v_heads_1BKD.shape}")
        keys_BKPD = self.layer_past[0]
        values_BKPD = self.layer_past[1]
        ttnn.experimental.paged_update_cache(keys_BKPD, k_heads_1BKD, update_idxs=[current_pos] * self.max_batch_size)
        ttnn.experimental.paged_update_cache(values_BKPD, v_heads_1BKD, update_idxs=[current_pos] * self.max_batch_size)
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        ###
        # Attention
        ###
        # print(f"q_heads_1BQD: {q_heads_1BQD.shape}")
        attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads_1BQD,
            keys_BKPD,
            values_BKPD,
            cur_pos=[current_pos] * self.max_batch_size,
            scale=self.attn_output_multiplier,
            attn_mask=attn_masks,
            is_causal=True if attn_masks is None else False,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                exp_approx_mode=False,
                q_chunk_size=256,
                k_chunk_size=256,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # FIXME: why not L1 height sharded
        )
        ttnn.deallocate(q_heads_1BQD)

        # print(f"attn_output_1G4D: {attn_output_1G4D.shape}")
        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=ttnn.create_sharded_memory_config(
                shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
                core_grid=ttnn.CoreRangeSet({num_to_corerange(self.max_batch_size)}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )
        ttnn.deallocate(attn_output_1G4D)

        # print(f"attn_output_1BQD: {attn_output_1BQD.shape}")
        attn_output_11BH = ttnn.experimental.nlp_concat_heads_decode(attn_output_11BH, num_heads=self.n_heads)

        # print(f"attn_output_11BH: {attn_output_11BH.shape}")
        attn_output_11BH = ttnn.matmul(
            attn_output_11BH,
            self.wo,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_attn,
        )

        return attn_output_11BH

    def forward_prefill(
        self,
        x,
        rot_mats,
        user_id,
        attn_masks,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[Tuple[ttnn.Tensor]]]:
        """
        x: (1, batch, seq_len, hidden_dim)
        current_pos: the length of the KV cache. Same as current token's index.
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        S : seq_len (8)
        H : dim (3072)
        D : head_dim (96)
        P : padded_layer_past_len
        """
        seq_len = x.shape[-2]

        ###
        # QKV Linear
        ###
        # print(f"x_11SH: {x.shape}")
        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)

        ###
        # Reshape and rotary embeddings TODO: Scaled ROPE for >4k Context length
        ###
        # print(f"fused_qkv: {xqkv_fused.shape}")
        # split qkv into heads
        (
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # TODO: Should this be removed one padding optimisations are in?
        if seq_len % 32 != 0:
            q_heads_1QSD = ttnn.slice(q_heads_1QSD, [0, 0, 0, 0], [1, self.n_kv_heads, seq_len, self.head_dim])
            k_heads_1KSD = ttnn.slice(k_heads_1KSD, [0, 0, 0, 0], [1, self.n_kv_heads, seq_len, self.head_dim])
            v_heads_1VSD = ttnn.slice(v_heads_1VSD, [0, 0, 0, 0], [1, self.n_kv_heads, seq_len, self.head_dim])

        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1QSD.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1KSD.memory_config()

        # print(f"q_heads_1QSD: {q_heads_1QSD.shape}")
        q_heads_1QSD = ttnn.transpose(q_heads_1QSD, 1, 2)
        q_heads_1QSD = ttnn.matmul(
            q_heads_1QSD,
            rot_mats,
            memory_config=self.q_mem_config,
            compute_kernel_config=self.compute_kernel_attn,
            # [bsz, seqlen, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [bsz, seqlen, padd_heads, head_dim]
        )
        q_heads_1QSD = ttnn.transpose(q_heads_1QSD, 1, 2)

        # print(f"k_heads_1KSD: {k_heads_1KSD.shape}")
        k_heads_1KSD = ttnn.transpose(k_heads_1KSD, 1, 2)
        k_heads_1KSD = ttnn.matmul(
            k_heads_1KSD,
            rot_mats,
            memory_config=self.k_mem_config,
            compute_kernel_config=self.compute_kernel_attn,
            # [bsz, seqlen, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [bsz, seqlen, padd_heads, head_dim]
        )
        k_heads_1KSD = ttnn.transpose(k_heads_1KSD, 1, 2)

        # print(f"q_heads_1QSD: {q_heads_1QSD.shape}")
        # print(f"k_heads_1KSD: {k_heads_1KSD.shape}")
        # q_heads_1QSD = ttnn.experimental.rotary_embedding(
        #     q_heads_1QSD,
        #     rot_mats[0],
        #     rot_mats[1],
        #     token_index=None,
        #     memory_config=self.q_mem_config,
        #     compute_kernel_config=self.compute_kernel_attn,
        # )
        # k_heads_1KSD = ttnn.experimental.rotary_embedding(
        #     k_heads_1KSD,
        #     rot_mats[0],
        #     rot_mats[1],
        #     token_index=None,
        #     memory_config=self.k_mem_config,
        #     compute_kernel_config=self.compute_kernel_attn,
        # )

        ###
        # KV update
        ###
        # print(f"v_heads_1VSD: {v_heads_1VSD.shape}")
        k_heads_1KSD = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        v_heads_1VSD = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
        keys_BKPD = self.layer_past[0]
        values_BKPD = self.layer_past[1]
        ttnn.fill_cache(keys_BKPD, k_heads_1KSD, user_id)
        ttnn.fill_cache(values_BKPD, v_heads_1VSD, user_id)

        ###
        # Attention
        ###
        # print(f"q_heads_1QSD: {q_heads_1QSD.shape}")
        # print(f"k_heads_1KDS: {k_heads_1KDS.shape}")
        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            attn_mask=attn_masks,
            is_causal=True if attn_masks is None else False,
            scale=self.attn_output_multiplier,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                exp_approx_mode=False,
                q_chunk_size=256,
                k_chunk_size=256,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(q_heads_1QSD)
        ttnn.deallocate(k_heads_1KSD)
        ttnn.deallocate(v_heads_1VSD)

        # print(f"attn_output_1QSD: {attn_output_1QSD.shape}")
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(attn_output_1QSD)
        ttnn.deallocate(attn_output_1QSD)

        # print(f"attn_output_11SH: {attn_output_11SH.shape}")
        attn_output_11SH = ttnn.matmul(
            attn_output_11SH,
            self.wo,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_attn,
        )

        return attn_output_11SH

    def forward(
        self,
        x,
        rot_mats,
        current_pos,
        attn_masks,
        mode: str = "decode",
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                current_pos,
                attn_masks,
            )
        elif mode == "decode":
            return self.forward_decode(
                x,
                rot_mats,
                current_pos,
                attn_masks,
            )
        else:
            raise ValueError("Invalid run mode")
