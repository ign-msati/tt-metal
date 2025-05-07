# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtPhi3MiniRotaryEmbedding(LightweightModule):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.device = device
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # self.register_buffer("inv_freq", None, persistent=False)
        # Setup Layer

    def forward(self, x: ttnn.Tensor, position_ids: ttnn.Tensor, seq_len=None) -> ttnn.Tensor:
        pass
