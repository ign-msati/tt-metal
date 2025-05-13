# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch
import ttnn
from models.utility_functions import pad_by_zero
from models.common.lightweightmodule import LightweightModule
from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.experimental.phi3_mini.tt.phi3_mini_mlp import TtPhi3MiniMLP
from models.helper_funcs import Linear
from models.experimental.phi3_mini.tt.phi3_mini_decoder import TtPhi3MiniDecoder
class TtPhi3MiniModel(LightweightModule):
    # def __init__(self, config, state_dict, base_address, device, layer_idx):
    def __init__(self, config, state_dict, device, vocab_size=51200, 
                 n_decoder=32, dim=2048, heads=32, rope_theta=10000.0
    ):
        super().__init__()
        ###############################333
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size
        self.hidden_size =config.hidden_size
        self.n_layer = config.num_hidden_layers
        #####################################
        self.embedding_tokens_weights = pad_by_zero(state_dict['model.embed_tokens.weight'], device)[0]
        
        # self.n_decoder = self.n_layer
        self.layers = []
        for i in range(self.n_layer):
            LAYER_INDEX = i
            base_address = f"model.layers.{LAYER_INDEX}"
            # Tt phi3-mini decoder_layers
            decoder_layers = TtPhi3MiniDecoder(
                config=config,
                base_address=base_address,
                device=device,
                state_dict=state_dict,
                layer_idx=LAYER_INDEX,
            )
            self.layers.append(decoder_layers)

        # self.final_layernorm_weights =state_dict['model.norm.weight']
        self.final_layernorm_weights = pad_by_zero(state_dict['model.norm.weight'], device)[0]
        # self.lm_head_weights=state_dict['lm_head.weight']
        # self.lm_head_weights= pad_by_zero(state_dict['lm_head.weight'], device)[0]
        # self.lm_head_weights  = ttnn.transpose( self.lm_head_weights, -2, -1)
        self.lm_head_weights= pad_by_zero(torch.transpose(state_dict['lm_head.weight'], -2, -1), device)[0]
        # self.lm_head_weights  = ttnn.transpose( self.lm_head_weights, -2, -1)

        self.variance_epsilon=config.rms_norm_eps
        # self.lm_head = Linear(
        #     3072,
        #     32064,
        #     self.lm_head_weights,
        #     None,
        # )

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        #     )
    """
    """

    def forward(
        self,
        inputs,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        ) -> Tuple[ttnn.Tensor]:
        # pass
        x = ttnn.embedding(inputs, self.embedding_tokens_weights)

        # for layer in self.layers:
        #     x = layer(x, position_ids=position_ids)

        tt_output_states=x
        # print("tt_output_statest_____",tt_output_states.shape)
        for layer in self.layers:
            tt_output_states = layer(tt_output_states, position_ids=position_ids)
            tt_output_states=tt_output_states[0]

        # x = ttnn.layer_norm(x, epsilon=1e-05, weight=self.final_layernorm_weights, bias=self.final_layernorm_bias)
        x = ttnn.rms_norm(tt_output_states, epsilon=self.variance_epsilon, weight=self.final_layernorm_weights)

        x = ttnn.linear(x, self.lm_head_weights)
        return x