# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn
import math

from models.common.lightweightmodule import LightweightModule
from models.helper_funcs import Linear
from models.utility_functions import pad_by_zero
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding


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
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[ttnn.Tensor] = None,
        output_attentions: bool = True,
        use_cache: bool = False,
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

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     pass

        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        # print(f"qkv: {qkv.shape}")
        # print(f"cos: {cos.shape}")
        # print(f"sin: {sin.shape}")
        # print(f"query_states: {query_states.shape}")
        # print(f"value_states: {value_states.shape}")

        # cos = ttnn.pad(cos, padding=((0, 0), (0, 0), (0, 32)), value=0)
        # sin = ttnn.pad(sin, padding=((0, 0), (0, 0), (0, 32)), value=0)
        # query_states = ttnn.pad(query_states, padding=((0, 0), (0, 0), (0, 0), (0, 32)), value=0)
        # value_states = ttnn.pad(value_states, padding=((0, 0), (0, 0), (0, 0), (0, 32)), value=0)

        # print(f"cos: {cos.shape}")
        # print(f"sin: {sin.shape}")
        # print(f"query_states: {query_states.shape}")
        # print(f"value_states: {value_states.shape}")

        # # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin)
        # key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin)

        if past_key_value is not None:
            pass

        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups) num_key_value_groups = 1
        # value_states = repeat_kv(value_states, self.num_key_value_groups) num_key_value_groups = 1

        key_states = ttnn.transpose(key_states, 2, 3)

        attn_weights = ttnn.matmul(query_states, key_states)
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
