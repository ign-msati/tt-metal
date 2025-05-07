# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding


class TtPhi3MiniLongRoPEScaledRotaryEmbedding(TtPhi3MiniRotaryEmbedding):
    def __init__(self, dim, config, device):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings
        # Setup Layer

    def forward(self, x: ttnn.Tensor, position_ids: ttnn.Tensor, seq_len=None) -> ttnn.Tensor:
        pass
