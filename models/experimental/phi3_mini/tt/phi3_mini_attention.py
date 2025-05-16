from typing import Optional, Tuple

import torch
import ttnn
import math

from models.common.lightweightmodule import LightweightModule

import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule
from models.experimental.phi3_mini_may.reference.rope import Phi3LongRoPEScaledRotaryEmbedding


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
    position = torch.LongTensor([[position_ids]])

    cos, sin = rope_scale(position, seq_len=None)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = ttnn.from_torch(
        query_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    key_states = ttnn.from_torch(
        key_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return query_states, key_states


class TtPhi3MiniAttention(LightweightModule):
    def __init__(self, config, state_dict, base_address, layer_idx, device, kernel_args):
        super().__init__()

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
        self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.model_config)

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
        B : batch_size (32)
        H : dim (6144)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        x_11BH = x
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[current_pos]
        attn_mask_1B4P = attn_masks
        batch = x.shape[-2]

        ###
        # QKV Linear
        ###

        print(f"x: {x.shape}")

        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)

        print(f"xqkv_fused: {xqkv_fused.shape}")
        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, batch, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
        print(f"xqkv_fused: {xqkv_fused.shape}")

        ###
        # Reshape and rotary embeddings TODO: Scaled ROPE
        ###
        (
            q_heads_1BQD,
            k_heads_1BKD,
            # q_heads_pre_rot_1BQD,
            # k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # print(f"q_heads_1BQD: {q_heads_1BQD.shape}")
        query_pos = self.n_heads * self.head_dim
        query_states = xqkv_fused[..., :query_pos]
        q_heads_BQ1D = ttnn.reshape(
            query_states, (batch, self.n_heads, 1, self.head_dim), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(query_states)
        # key_states = xqkv_fused[..., query_pos : query_pos + self.n_kv_heads * self.head_dim]
        # k_heads_1BKD = ttnn.reshape(key_states, (1, batch, self.n_kv_heads, self.head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        # ttnn.deallocate(key_states)
        # value_states = xqkv_fused[..., query_pos + self.n_kv_heads * self.head_dim :]
        # v_heads_1BKD = ttnn.reshape(value_states, (1, batch, self.n_kv_heads, self.head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        # ttnn.deallocate(value_states)

        ttnn.deallocate(q_heads_1BQD)
        ttnn.deallocate(xqkv_fused)
        print(f"k_heads_1BKD: {k_heads_1BKD.shape}")
        print(f"v_heads_1BKD: {v_heads_1BKD.shape}")

        # print(f"q_heads_1BQD layout: {q_heads_1BQD.layout}")
        # print(f"k_heads_1BKD layout: {k_heads_1BKD.layout}")
        # print(f"q_heads_1BQD mem: {q_heads_1BQD.memory_config}")
        # print(f"k_heads_1BKD mem: {k_heads_1BKD.memory_config}")

        # q_heads_1BQD, k_heads_1BKD = fall_back_torch_rope(
        #     q_heads_1BQD, k_heads_1BKD, self.rotary_emb, current_pos, self.mesh_device
        # )

        # print(f"q_heads_1BQD layout: {q_heads_1BQD.layout}")
        # print(f"k_heads_1BKD layout: {k_heads_1BKD.layout}")
        # print(f"q_heads_1BQD mem: {q_heads_1BQD.memory_config}")
        # print(f"k_heads_1BKD mem: {k_heads_1BKD.memory_config}")

        ###
        # KV update
        ###
        keys = self.layer_past[0]
        values = self.layer_past[1]

        # k_heads, [seqlen, bsz, n_kv_heads, head_dim]
        # v_heads [seqlen, bsz, n_kv_heads, head_dim]
        # keys, [max_batch_size, n_kv_heads, max_seq_len, head_dim]
        ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs=[current_pos] * batch)
        ttnn.experimental.paged_update_cache(values, v_heads_1BKD, update_idxs=[current_pos] * batch)
        # ttnn.update_cache(keys, k_heads_1BKD, current_pos)
        # ttnn.update_cache(values, v_heads_1BKD, current_pos)
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        values = self.layer_past[1]
        print(f"keys: {keys.shape}")
        print(f"values: {values.shape}")
        ###
        # Attention
        ###
        # transpose keys
        keys_BKD1 = ttnn.transpose(keys, 2, 3)
        print(f"pre_softmax_matmul_q_heads_BQ1D: {q_heads_BQ1D.shape}")
        print(f"pre_softmax_keys_BKD1: {keys_BKD1.shape}")
        wattn_1BQS = ttnn.matmul(
            q_heads_BQ1D,
            keys_BKD1,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_attn,
        )
        # ttnn.deallocate(q_heads_1BQD)
        ttnn.deallocate(keys_BKD1)

        print(f"wattn_1BQS: {wattn_1BQS.shape}")

        # wattn_1BQS = ttnn.mul(
        #     wattn_1BQS, self.attn_output_multiplier, memory_config=ttnn.L1_MEMORY_CONFIG
        # )
        # if attn_mask_1B4P is not None:
        #     wattn_1BQS = ttnn.add(
        #         wattn_1BQS,
        #         attn_mask_1B4P,
        #         memory_config=ttnn.L1_MEMORY_CONFIG,
        #     )
        # wattn_1BQS = ttnn.softmax(wattn_1BQS, dim=-1)
        wattn_1BQS = ttnn.scale_mask_softmax_in_place(
            wattn_1BQS,
            scale=self.attn_output_multiplier,
            mask=attn_mask_1B4P,
            is_causal_mask=False,  # causal_mask=False will broadcast attention mask across all heads
        )

        print(f"wattn_1BQS_softmax: {wattn_1BQS.shape}")

        attn_O = ttnn.matmul(
            wattn_1BQS,
            values,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_attn,
        )

        print(f"attn_O: {attn_O.shape}")
        attn_O = ttnn.transpose(attn_O, 1, 2)
        attn_O = ttnn.reshape(attn_O, (1, batch, self.hidden_size))

        attn_O = ttnn.matmul(
            attn_O,
            self.wo,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_attn,
        )
        print(f"attn_O: {attn_O.shape}")

        return attn_O
