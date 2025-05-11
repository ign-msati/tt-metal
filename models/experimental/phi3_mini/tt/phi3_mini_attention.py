# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
import math

from models.common.lightweightmodule import LightweightModule
from models.helper_funcs import Linear
from models.utility_functions import pad_by_zero
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding
from models.experimental.phi3_mini.reference.rope import Phi3LongRoPEScaledRotaryEmbedding


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
    def __init__(self, config, state_dict, base_address, layer_idx, device):
        super().__init__()

        self.device = device
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)

        # Get the weights
        self.o_proj_weight = pad_by_zero(state_dict[f"{base_address}.o_proj.weight"], self.device)[0]
        self.qkv_proj_weight = pad_by_zero(state_dict[f"{base_address}.qkv_proj.weight"], self.device)[0]

        # Setup Layers
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            self.o_proj_weight,
            None,
        )
        self.qkv_proj = Linear(
            self.hidden_size,
            op_size,
            self.qkv_proj_weight,
            None,
        )

        self._init_rope()
        rotary_dim = config.hidden_size // config.num_attention_heads
        self.torch_rope_scale = Phi3LongRoPEScaledRotaryEmbedding(rotary_dim, config)

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
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor | torch.Tensor] = None,
        past_key_value: Optional[ttnn.Tensor] = None,
        output_attentions: bool = True,
        use_cache: bool = False,
        fall_back_to_torch: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[Tuple[ttnn.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = ttnn.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim))

        query_states = ttnn.transpose(query_states, 1, 2)
        key_states = ttnn.transpose(key_states, 1, 2)
        value_states = ttnn.transpose(value_states, 1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            pass

        if fall_back_to_torch:
            query_states, key_states = fall_back_torch_rope(
                query_states, key_states, self.torch_rope_scale, position_ids, self.device
            )
        else:
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
            cos = ttnn.unsqueeze(cos, 1)
            sin = ttnn.unsqueeze(sin, 1)

            neg_half = (-1) * query_states[..., query_states.shape[-1] // 2 :]
            pos_half = query_states[..., : query_states.shape[-1] // 2]
            rotated_query_states = ttnn.concat([neg_half, pos_half], dim=-1)

            neg_half = (-1) * key_states[..., key_states.shape[-1] // 2 :]
            pos_half = key_states[..., : key_states.shape[-1] // 2]
            rotated_key_states = ttnn.concat([neg_half, pos_half], dim=-1)

            query_states = (query_states * cos) + (rotated_query_states * sin)
            key_states = (key_states * cos) + (rotated_key_states * sin)

        if past_key_value is not None:
            pass

        key_states = ttnn.transpose(key_states, 2, 3)

        attn_weights = ttnn.matmul(query_states, key_states, dtype=ttnn.bfloat16)
        attn_weights = attn_weights * (1.0 / math.sqrt(self.head_dim))

        if attention_mask is not None:
            pass

        # TODO: upcast attention to fp32
        attn_weights = ttnn.softmax(attn_weights, dim=-1)

        attn_output = ttnn.matmul(attn_weights, value_states)

        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
