# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn

from models.common.lightweightmodule import LightweightModule


class TtPhi3MiniDecoder(LightweightModule):
    def __init__(self, config, state_dict, base_address, device, layer_idx):
        super().__init__()
        pass

    """
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor]:
        pass
