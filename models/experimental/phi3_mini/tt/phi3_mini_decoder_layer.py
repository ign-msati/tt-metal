# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch
import ttnn
from models.utility_functions import pad_by_zero
from models.common.lightweightmodule import LightweightModule

from models.experimental.phi3_mini.tt.phi3_mini_decoder import TtPhi3MiniDecoder
class TtPhi3MiniDecoderLayer(LightweightModule):
    # def __init__(self, config, state_dict, base_address, device, layer_idx):
    def __init__(self, config, state_dict, device, LAYER_INDEX_SPLIT
    ):
        super().__init__()
        ###############################333
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size
        self.hidden_size =config.hidden_size
        self.n_layer = config.num_hidden_layers
        #####################################
   
        # self.n_decoder = self.n_layer
        self.layers = []
        for i in range(LAYER_INDEX_SPLIT):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        # ) -> Tuple[ttnn.Tensor]:
        # pass
        tt_output_states=hidden_states
        for layer in self.layers:
            tt_output_states = layer(tt_output_states, position_ids=position_ids)
            tt_output_states=tt_output_states[0]

        return tt_output_states