# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding


class TtPhi3MiniLongRoPEScaledRotaryEmbedding(TtPhi3MiniRotaryEmbedding):
    def __init__(self, dim, config, device):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = torch.tensor(config.rope_scaling["short_factor"])
        self.long_factor = torch.tensor(config.rope_scaling["long_factor"])
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.inv_freq_shape = torch.arange(0, self.dim, 2)

        self.inv_freq_long = 1.0 / (self.long_factor_factor * self.base**self.inv_freq_shape)
        self.inv_freq_short = 1.0 / (self.short_factor * self.base**self.inv_freq_shape)

        self.inv_freq_long_expanded = ttnn.from_torch(
            self.inv_freq_long[None, :, None].float().expand(self.original_max_position_embeddings, -1, 1)
        )
        self.inv_freq_short_expanded = ttnn.from_torch(
            self.inv_freq_short[None, :, None].float().expand(self.original_max_position_embeddings, -1, 1)
        )

    def forward(self, x: ttnn.Tensor, position_ids: ttnn.Tensor, seq_len=None) -> ttnn.Tensor:
        position_ids_expanded = position_ids[:, None, :]
        if seq_len > self.original_max_position_embeddings:
            freqs = ttnn.matmul(self.inv_freq_expanded_long[: len(position_ids)], position_ids_expanded)
        else:
            freqs = ttnn.matmul(self.inv_freq_expanded_short[: len(position_ids)], position_ids_expanded)

        freqs = ttnn.transpose(1, 2)

        emb = ttnn.concat([freqs, freqs], dim=-1)

        ttnn.deallocate(freqs)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        self.cos = emb.cos() * scaling_factor
        self.sin = emb.sin() * scaling_factor

        ttnn.deallocate(emb)

        return self.cos, self.sin
