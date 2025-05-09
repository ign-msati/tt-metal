# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch
import ttnn
from models.utility_functions import pad_by_zero
from models.common.lightweightmodule import LightweightModule
from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.experimental.phi3_mini.tt.phi3_mini_mlp import TtPhi3MiniMLP
class TtPhi3MiniDecoder(LightweightModule):
    def __init__(self, config, state_dict, base_address, device, layer_idx):
        super().__init__()

            # Tt phi3-mini attn layer
        SELF_LAYER_INDEX=0
        self.attention = TtPhi3MiniAttention(
            config=config,
            base_address=f"{base_address}.self_attn",
            # base_address=f"model.layers.{SELF_LAYER_INDEX}.self_attn",
            device=device,
            state_dict=state_dict,
            layer_idx=SELF_LAYER_INDEX,
        )
        self.feed_forward = TtPhi3MiniMLP(
            config=config,
            base_address=f"{base_address}.mlp",
            # base_address=f"model.layers.{SELF_LAYER_INDEX}.mlp",
            device=device,
            state_dict=state_dict,
            )
        # f"model.layers.{SELF_LAYER_INDEX}.input_layernorm.weight"
        # state_dict['model.layers.0.input_layernorm.weight']
        # base_address=f"model.layers.{SELF_LAYER_INDEX}"
        # state_dict['model.layers.0.input_layernorm.weight']
        # state_dict['model.layers.0.post_attention_layernorm.weight']
        self.input_layernorm = pad_by_zero(state_dict[f"{base_address}.input_layernorm.weight"], device)[0]
        self.post_attention_layernorm = pad_by_zero(state_dict[f"{base_address}.post_attention_layernorm.weight"], device)[0]
        # self.layernorm_weights = f"model.layers.{SELF_LAYER_INDEX}.input_layernorm.weight"
        # self.layernorm_bias = layernorm_bias
        self.variance_epsilon=config.rms_norm_eps


    """
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        
        ) -> Tuple[ttnn.Tensor]:
        # pass
        residual = hidden_states

        # hidden_states = ttnn.layer_norm(hidden_states, self.layernorm_weights)
        hidden_states = ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.input_layernorm)


        hidden_states = self.attention(hidden_states, position_ids=position_ids)

        # hidden_states = residual + hidden_states 

        residual = hidden_states

        # hidden_states = ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.post_attention_layernorm)

        hidden_states = self.feed_forward(hidden_states)

        # # hidden_states = residual + hidden_states
        return hidden_states

        # ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.weight)