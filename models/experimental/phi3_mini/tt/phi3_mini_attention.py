from typing import Optional, Tuple

import torch
import ttnn
import math

from models.common.lightweightmodule import LightweightModule
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def fall_back_torch_rope(query_states, key_states, rope_scale, position_ids, device):
    query_states = ttnn.to_torch(query_states)
    key_states = ttnn.to_torch(key_states)

    cos, sin = rope_scale(position_ids, seq_len=None)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = ttnn.from_torch(
        query_states,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    key_states = ttnn.from_torch(
        key_states,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return query_states, key_states


class TtPhi3MiniAttention(LightweightModule):
    def __init__(self, config, state_dict, base_address, layer_idx, device, kernel_args):
        super().__init__()

        self.num_devices = 1
        self.state_dict = state_dict
        self.mesh_device = device
        self.model_args = kernel_args

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
        cache_name = lambda _: None

        # self.o_proj_weight = pad_by_zero(state_dict[f"{base_address}.o_proj.weight"], self.device, tt_dtype=ttnn.bfloat8_b)[0]
        # self.qkv_proj_weight = pad_by_zero(state_dict[f"{base_address}.qkv_proj.weight"], self.device, tt_dtype=ttnn.bfloat8_b)[0]

        self.wqkv = ttnn.as_tensor(
            torch.transpose(
                state_dict[f"{base_address}.qkv_proj.weight"],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            # self.qkv_proj_weight, # Might have to unsqueeze
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_mem_configs["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
            # cache_file_name=cache_name(f"wqkv_multidevice_4d"), # Not needed for us, since out weights are cached by HF
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                state_dict[f"{base_address}.o_proj.weight"],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            # self.o_proj_weight, # Might need to unsqueeze
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_mem_configs["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
            # cache_file_name=cache_name(f"wo_multidevice4d_H"), # Same reason as above
        )

        cache_k = torch.zeros(  # 12582912000 * 1 bytes values -> 24GB :(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.as_tensor(
                lp,
                device=self.mesh_device,
                mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.bfloat8_b,
                layout=self.model_mem_configs["ATTN_W_LAYOUT_TILE"],
                memory_config=self.model_mem_configs["ATTN_CACHE_WEIGHTS_MEMCFG"],
                # cache_file_name=cache_name(f"empty_attn_cache_{cache_k.shape}"),
            )
            for lp in layer_past
        ]

        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

        # Will be filled during the initial warmup run
        self.q_mem_config = None
        self.k_mem_config = None

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = TtPhi3MiniRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                device=self.device,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = TtPhi3MiniLongRoPEScaledRotaryEmbedding(self.head_dim, self.config, self.device)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
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
        B : batch_size (32)
        H : dim (6144)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = min(nearest_32(current_pos + 1), self.max_seq_len)

        x_11BH = hidden_states
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[current_pos]
        attn_mask_1B4P = attn_masks
        ###
        # QKV matmuls
        ###

        xqkv_fused = ttnn.matmul(
            x_11BH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # program_config=self.model_mem_configs["QKV_MM_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        # split qkv into heads
        (
            q_heads_1B4D,
            k_heads_1B1D,
            v_heads_1B1D,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_mem_configs["HEIGHT_SHARDED_MEMCFG"],
        )
        xqkv_fused.deallocate(True)
        # new_key_states = ttnn.to_torch(k_heads_1B1D, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))

        ###
        # Rotary embeddings
        ###
        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1B4D.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1B1D.memory_config()

        q_heads_1B4D = ttnn.matmul(
            q_heads_1B4D,
            rot_mat,
            # program_config=self.model_mem_configs["ROT_MAT_MM_PROGCFG"], # might be 96 from 3072 // 32 TODO: fix config file ~nv
            memory_config=self.q_mem_config,
            compute_kernel_config=self.model_mem_configs["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )
        k_heads_1B1D = ttnn.matmul(
            k_heads_1B1D,
            rot_mat,
            # program_config=self.model_mem_configs["ROT_MAT_MM_PROGCFG"],
            memory_config=self.k_mem_config,
            compute_kernel_config=self.model_mem_configs["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )
        # rotmat_key_states = ttnn.to_torch(k_heads_1B1D, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))
        # rotmat = ttnn.to_torch(rot_mat, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))[0]

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        # TODO: Findout what is wrong and add back in
        # ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, current_pos)
        # ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, current_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        # TODO: Cache fix
        # keys_1BPD = ttnn.experimental.nlp_kv_cache_load_slice(
        #     keys_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        # )

        # query_states = ttnn.to_torch(q_heads_1B4D, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-2))
        # key_states = ttnn.to_torch(keys_1BPD, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.transpose(
            keys_1BPD,
            -2,
            -1,
            memory_config=self.model_mem_configs["HEIGHT_SHARDED_MEMCFG"],
        )
        keys_1BPD.deallocate(True)

        # scores matmul
        attn_1B4P_memconfig = self.model_mem_configs["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len)
        attn_1B4P = ttnn.matmul(
            q_heads_1B4D,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            program_config=self.model_mem_configs["SCORES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            memory_config=attn_1B4P_memconfig,
            compute_kernel_config=self.compute_kernel_attn,
        )
        q_heads_1B4D.deallocate(True)
        keys_1BDP.deallocate(True)

        # Softmax and scaling
        # FIXME: Maintain sharded memory layout when #9773 is fixed
        attn_1B4P = ttnn.sharded_to_interleaved(attn_1B4P, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_1B4P = attn_1B4P * self.attn_output_multiplier
        attn_1B4P = ttnn.interleaved_to_sharded(attn_1B4P, attn_1B4P_memconfig)

        attn_1B4P = ttnn.scale_mask_softmax_in_place(
            attn_1B4P,
            1.0,
            attn_mask_1B4P,
            program_config=self.model_mem_configs["ATTN_BATCHED_SOFTMAX_PROGCFG"](padded_layer_past_len),
            is_causal_mask=True,
        )
        # post_softmax = ttnn.to_torch(attn_1B4P, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=-2))[0]

        # values matmul
        values_1BPD = ttnn.experimental.nlp_kv_cache_load_slice(
            values_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        # value_states = ttnn.to_torch(values_1BPD, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))
        # x = ttnn.to_torch(x_11BH, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))[0]
        attn_output_1B4D = ttnn.matmul(
            attn_1B4P,
            values_1BPD,
            dtype=ttnn.bfloat16,
            memory_config=self.model_mem_configs["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            program_config=self.model_mem_configs["VALUES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            compute_kernel_config=self.compute_kernel_attn,
        )
        attn_1B4P.deallocate(True)
        values_1BPD.deallocate(True)

        # value_output = ttnn.to_torch(attn_output_1B4D, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))[0]

        attn_output_11BH = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=6,
        )
        attn_output_1B4D.deallocate(True)

        attn_output_11BH = ttnn.sharded_to_interleaved(attn_output_11BH, memory_config=ttnn.L1_MEMORY_CONFIG)

        ###
        # Output matmul
        ###
        # All gather
        dense_outputs_11BH_gathered = ttnn.all_gather(attn_output_11BH, dim=3, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(
            dense_outputs_11BH_gathered,
            wo,
            memory_config=self.model_mem_configs["LM_HEAD_OUTPUT_MEMCFG"],
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_mem_configs["LM_HEAD_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )

        # attn_output = ttnn.to_torch(dense_outputs_11BH, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))[0]
        # attn_mask = ttnn.to_torch(attn_mask_1B4P, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=0))[0]

        dense_outputs_11BH_gathered.deallocate(True)
        return dense_outputs_11BH
