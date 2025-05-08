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
        self.inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim

        # Both these are vectors of length 48
        self.inv_freq_long = 1.0 / (self.long_factor * self.base**self.inv_freq_shape)
        self.inv_freq_short = 1.0 / (self.short_factor * self.base**self.inv_freq_shape)

        # These are matrices of shape (131072, 48, 1)
        # self.inv_freq_long_expanded = ttnn.from_torch(self.inv_freq_long[None, :, None].float().expand(self.original_max_position_embeddings, -1, 1))
        # self.inv_freq_short_expanded = ttnn.from_torch(self.inv_freq_short[None, :, None].float().expand(self.original_max_position_embeddings, -1, 1))

    def forward(self, x: ttnn.Tensor, position_ids: ttnn.Tensor, seq_len=None) -> ttnn.Tensor:
        # position_ids_expanded = position_ids[:, None, :] # Our assumption about the shape of position id seems to be wrong
        if seq_len > self.original_max_position_embeddings:
            self.inv_freq_long_expanded = ttnn.from_torch(
                self.inv_freq_long[None, :, None].float().expand(position_ids.shape[0], -1, 1),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
            )
            freqs = ttnn.matmul(self.inv_freq_long_expanded, position_ids, dtype=ttnn.float32)
            ttnn.deallocate(self.inv_freq_long_expanded)
        else:
            self.inv_freq_short_expanded = ttnn.from_torch(
                self.inv_freq_short[None, :, None].float().expand(position_ids.shape[0], -1, 1),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
            )
            freqs = ttnn.matmul(self.inv_freq_short_expanded, position_ids, dtype=ttnn.float32)
            ttnn.deallocate(self.inv_freq_short_expanded)

        freqs = ttnn.transpose(freqs, 1, 2)
        emb = ttnn.concat([freqs, freqs], dim=-1)

        ttnn.deallocate(freqs)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        self.cos = ttnn.cos(emb) * scaling_factor
        self.sin = ttnn.sin(emb) * scaling_factor

        ttnn.deallocate(emb)

        return self.cos, self.sin
