# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch
import ttnn
from models.utility_functions import pad_by_zero
from models.common.lightweightmodule import LightweightModule
from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.experimental.phi3_mini.tt.phi3_mini_mlp import TtPhi3MiniMLP
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.phi3_mini.tt.phi3_mini_decoder import TtPhi3MiniDecoder
class TtPhi3MiniEmbedding(LightweightModule):
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
        # self.embedding_tokens_weights = state_dict['model.embed_tokens.weight']

        self.embedding_tokens_weights = pad_by_zero(state_dict['model.embed_tokens.weight'], device)[0]

 

    def forward(
        self,
        inputs,
        ) -> Tuple[ttnn.Tensor]:
        # pass
        x = ttnn.embedding(inputs, self.embedding_tokens_weights)

 
        return x