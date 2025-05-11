# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
from models.experimental.phi3_mini.tt.phi3_mini_rotary_embedding import TtPhi3MiniRotaryEmbedding


class TtPhi3MiniLongRoPEScaledRotaryEmbedding(TtPhi3MiniRotaryEmbedding):
    def __init__(self, dim, config, device):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32)
        self.long_factor = torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32)
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.inv_freq_shape = torch.arange(0, self.dim, 2).float() / self.dim

        self.inv_freq_long = 1.0 / (self.long_factor * self.base**self.inv_freq_shape)
        self.inv_freq_short = 1.0 / (self.short_factor * self.base**self.inv_freq_shape)
        self.scale = self.max_position_embeddings / self.original_max_position_embeddings
        if self.scale <= 1.0:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = math.sqrt(1 + math.log(self.scale) / math.log(self.original_max_position_embeddings))

    def forward(self, x: ttnn.Tensor, position_ids: ttnn.Tensor, seq_len=None) -> ttnn.Tensor:
        if seq_len > self.original_max_position_embeddings:
            self.inv_freq_expanded = ttnn.from_torch(
                self.inv_freq_long[None, :, None].float().expand(position_ids.shape[0], -1, 1),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
            )
        else:
            self.inv_freq_expanded = ttnn.from_torch(
                self.inv_freq_short[None, :, None].float().expand(position_ids.shape[0], -1, 1),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
            )

        position_ids = ttnn.unsqueeze(position_ids, 1)
        freqs = ttnn.matmul(self.inv_freq_expanded, position_ids, dtype=ttnn.float32)

        freqs = ttnn.transpose(freqs, 1, 2)
        emb = ttnn.concat([freqs, freqs], dim=-1)

        self.cos = ttnn.cos(emb) * self.scaling_factor
        self.sin = ttnn.sin(emb) * self.scaling_factor

        return self.cos, self.sin
