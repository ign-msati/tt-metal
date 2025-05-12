# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.common.lightweightmodule import LightweightModule


# class TtPhi3MiniAttention(LightweightModule):
#     def __init__(self, config, state_dict, base_address, device, layer_idx):
#         super().__init__()
#         pass

#     """
#     """

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#     ) -> Tuple[ttnn.Tensor]:
#         pass


#####################3
import ttnn
from models.common.lightweightmodule import LightweightModule

# from models.experimental.phi_15.tt.phi_rotary_embedding import PhiRotaryEmbedding
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding
from models.experimental.phi3_mini.tt.phi3_mini_attention import fall_back_torch_rope
from models.experimental.phi3_mini.reference.rope import Phi3LongRoPEScaledRotaryEmbedding
class TtPhi3MiniAttention(LightweightModule):
    def __init__(self, device, config, parameters, layer_idx):
        super().__init__()

        # self.fused_qkv_weight = ttnn.concat(
        #     [parameters.q_proj.weight, parameters.k_proj.weight, parameters.v_proj.weight], dim=-1
        # )
        self.fused_qkv_weight = parameters['qkv_proj'].weight

        # self.fused_qkv_bias = ttnn.concat(
        #     [parameters.q_proj.bias, parameters.k_proj.bias, parameters.v_proj.bias],
        #     dim=-1,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        # )

        self.dense_weight = parameters['o_proj'].weight

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        # self.dense_bias = ttnn.to_device(parameters.dense.bias, device, ttnn.L1_MEMORY_CONFIG)
        # self.rotary_embedding = TtPhi3MiniLongRoPEScaledRotaryEmbedding(self.head_dim,device, config)
        rotary_dim = config.hidden_size // config.num_attention_heads
        self.torch_rope_scale = Phi3LongRoPEScaledRotaryEmbedding(rotary_dim, config)
        self.layer_idx = layer_idx
        self.device=device

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        past_key_values: ttnn.Tensor,
        use_cache: bool = False,
        fall_back_to_torch=True
    ):
        batch, seq_len, hidden_size = hidden_states.shape

        assert hidden_size == self.hidden_dim

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        fused_qkv_output = ttnn.linear(
            hidden_states,
            self.fused_qkv_weight,
            # bias=self.fused_qkv_bias,
            # bias=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        # return fused_qkv_output
        (
            query,
            key,
            value,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            transpose_key=False,
            num_heads=self.num_heads,
        )
        ttnn.deallocate(fused_qkv_output)
        # fall_back_to_torch=True
        if fall_back_to_torch:
            query, key = fall_back_torch_rope(
                query, key, self.torch_rope_scale, position_ids, self.device
            )
        # # query = self.rotary_embedding(query)
        # # key = self.rotary_embedding(key)
        key = ttnn.transpose(key, 2, 3)

        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        attention_probs = ttnn.transformer.attention_softmax_(
            attention_scores, attention_mask=attention_mask, head_size=self.head_dim
        )

        context_layer = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(attention_probs)

        context_layer_after_concatenate_heads = ttnn.transformer.concatenate_heads(
            context_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(context_layer)
        # print(f"ouput is {context_layer_after_concatenate_heads}")

        self_output = ttnn.linear(
            context_layer_after_concatenate_heads,
            self.dense_weight,
            # bias=False,
            # bias=self.dense_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(context_layer_after_concatenate_heads)

        return self_output, past_key_values


##########################
