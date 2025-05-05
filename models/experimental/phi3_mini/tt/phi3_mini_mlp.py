# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn

from models.common.lightweightmodule import LightweightModule


class TtPhi3MiniMLP(LightweightModule):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        pass

    """
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[ttnn.Tensor]:
        pass
