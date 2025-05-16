from typing import Optional, Tuple

import torch
import ttnn
import math

from models.common.lightweightmodule import LightweightModule
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule


class TtPhi3MiniAttention(LightweightModule):
    def __init__(self, config, state_dict, base_address, layer_idx, device, kernel_args):
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
        self.max_seq_len = config.max_position_embeddings // 128
        self.max_batch_size = 8
        self.n_kv_heads = config.num_key_value_heads
        self.attn_output_multiplier = 1.0 / math.sqrt(self.head_dim)

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.model_mem_configs = self.model_args.get_model_mem_configs()
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
                dtype=ttnn.bfloat16,
                # dtype=ttnn.bfloat8_b,
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

    def forward(
        self,
        x: ttnn.Tensor,
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
        rot_mat = rot_mats[current_pos]
        attn_mask_B11P = attn_masks
        batch = x.shape[-2]

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

        # Reshape such that true unpadded batch is tracked in shape
        # TODO: check what this does and is it needed
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, batch, xqkv_fused.shape[3]), (1, 1, 32, xqkv_fused.shape[3]))

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
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads_1BQD)

        # TODO: Check if this can be substituted by a normal transpose
        query_pos = self.n_heads * self.head_dim
        query_states = xqkv_fused[..., :query_pos]
        q_heads_B1QD = ttnn.reshape(
            query_states, (batch, 1, self.n_kv_heads, self.head_dim), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(query_states)
        ttnn.deallocate(xqkv_fused)

        if self.q_mem_config is None:
            self.q_mem_config = q_heads_B1QD.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1BKD.memory_config()

        # print(f"q_heads_B1QD: {q_heads_B1QD.shape}")
        q_heads_B1QD = ttnn.matmul(
            q_heads_B1QD,
            rot_mat,
            memory_config=self.q_mem_config,
            compute_kernel_config=self.compute_kernel_attn
            # [bsz, seqlen, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [bsz, seqlen, padd_heads, head_dim]
        )
        # print(f"k_heads_1BKD: {k_heads_1BKD.shape}")
        k_heads_1BKD = ttnn.matmul(
            k_heads_1BKD,
            rot_mat,
            memory_config=self.k_mem_config,
            compute_kernel_config=self.compute_kernel_attn
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )

        ###
        # KV update
        ###
        # print(f"v_heads_1BKD: {v_heads_1BKD.shape}")
        keys_BKPD = self.layer_past[0]
        values_BKPD = self.layer_past[1]
        ttnn.experimental.paged_update_cache(keys_BKPD, k_heads_1BKD, update_idxs=[current_pos] * batch)
        ttnn.experimental.paged_update_cache(values_BKPD, v_heads_1BKD, update_idxs=[current_pos] * batch)
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        ###
        # Attention
        ###
        # transpose keys and queries
        q_heads_BQ1D = ttnn.transpose(q_heads_B1QD, 1, 2)
        ttnn.deallocate(q_heads_B1QD)
        # print(f"keys_BKPD: {keys_BKPD.shape}")
        keys_BKDP = ttnn.transpose(keys_BKPD, 2, 3)
        # print(f"q_heads_BQ1D: {q_heads_BQ1D.shape}")
        # print(f"keys_BKDP: {keys_BKDP.shape}")
        wattn_BQ1P = ttnn.matmul(
            q_heads_BQ1D,
            keys_BKDP,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_attn,
        )
        ttnn.deallocate(q_heads_BQ1D)
        ttnn.deallocate(keys_BKDP)

        # TODO: Load nearest32 padded attention mask instead max_seqlen
        # print(f"wattn_BQ1P: {wattn_BQ1P.shape}")
        wattn_BQ1P = ttnn.scale_mask_softmax_in_place(
            wattn_BQ1P,
            scale=self.attn_output_multiplier,
            mask=attn_mask_B11P,
            is_causal_mask=False,  # causal_mask=False will broadcast attention mask across all heads
        )

        # print(f"values_BKPD: {values_BKPD.shape}")
        attn_output_BQ1D = ttnn.matmul(
            wattn_BQ1P,
            values_BKPD,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_attn,
        )
        ttnn.deallocate(wattn_BQ1P)

        # print(f"attn_output_BQ1D: {attn_output_BQ1D.shape}")
        attn_output_B1QD = ttnn.transpose(attn_output_BQ1D, 1, 2)
        ttnn.deallocate(attn_output_BQ1D)

        # print(f"attn_output_B1QD: {attn_output_B1QD.shape}")
        attn_output_11BH = ttnn.reshape(attn_output_B1QD, (1, batch, self.hidden_size))
        ttnn.deallocate(attn_output_B1QD)

        # print(f"attn_output_11BH: {attn_output_11BH.shape}")
        attn_output_11BH = ttnn.matmul(
            attn_output_11BH,
            self.wo,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_attn,
        )

        return attn_output_11BH
