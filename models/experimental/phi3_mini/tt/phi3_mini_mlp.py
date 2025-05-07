# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.helper_funcs import Linear
from models.utility_functions import pad_by_zero
from models.common.lightweightmodule import LightweightModule


def _get_chunks(input_tensor: ttnn.Tensor, num_chunks: int, num_features: int):
    slice_index = math.ceil(num_features / num_chunks)
    return input_tensor[..., :slice_index], input_tensor[..., slice_index:]


class TtPhi3MiniMLP(LightweightModule):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.device = device
        self.config = config

        # Get the weights
        self.gate_up_proj_weight = pad_by_zero(state_dict[f"{base_address}.gate_up_proj.weight"], self.device)[0]
        self.down_proj_weight = pad_by_zero(state_dict[f"{base_address}.down_proj.weight"], self.device)[0]

        self.input_dim = self.gate_up_proj_weight.padded_shape[-1]
        self.hidden_dim = self.down_proj_weight.padded_shape[-1]

        # Setup Layers
        self.gate_up_proj = Linear(
            self.input_dim,
            2 * self.hidden_dim,
            self.gate_up_proj_weight,
            None,
        )
        self.down_proj = Linear(
            self.hidden_dim,
            self.input_dim,
            self.down_proj_weight,
            None,
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = _get_chunks(up_states, 2, num_features=2 * self.hidden_dim)

        up_states = up_states * ttnn.silu(gate)

        return self.down_proj(up_states)
