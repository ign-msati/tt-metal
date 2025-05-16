# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh
from models.utility_functions import nearest_32


def precompute_freqs(dim: int, end: int, ext_scale_factor: torch.Tensor, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions, grok-style.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Grok-1 uses 10000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (ext_scale_factor * theta ** (torch.arange(0, dim, 2).float() / dim))
    scalar = 1.1902380714238083  # TODO: calculate from model config instead of hardcoding
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.cos(emb) * scalar, torch.sin(emb) * scalar
    return cos, sin


def freq_row_to_rotation_matrix(cos_row, sin_row):
    """
    Transform cos/sin frequency rows to a dim x dim rotation matrix
    that implements cos + rotate_half * sin
    """

    d = len(sin_row)
    m_cos = torch.diag(cos_row)
    m_sin = torch.diag(sin_row)
    d = len(sin_row)
    m_rot_sin = torch.cat([m_sin[d // 2 :], -m_sin[: d // 2]])
    return m_cos + m_rot_sin


def get_rotation_mat(dhead, end, ext_scale_factor):
    cos, sin = precompute_freqs(dhead, end, ext_scale_factor)
    rot_mat = [freq_row_to_rotation_matrix(c, s) for c, s in zip(cos, sin)]
    return rot_mat


def prepare_inputs_ttnn(x_bsh, hidden_size, current_pos, mesh_device, max_seq_len):
    """
    Prepare inputs for decode mode.
    x: (batch, seq, hidden_dim)
    B: batch (32)
    S: sequence len (1)
    H: dim (4096)
    """
    assert x_bsh.size(2) == hidden_size
    assert len(x_bsh.size()) == 3

    batch = x_bsh.size(0)
    seq_len = x_bsh.size(1)
    assert seq_len == 1, "Only supporting decode mode"

    x_1SBH = x_bsh.view(1, seq_len, batch, hidden_size)

    # input goes to L1
    xs_1SBH = ttnn.from_torch(
        x_1SBH,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Attention mask
    padded_layer_past_len = nearest_32(current_pos + 1)
    attn_mask = torch.zeros(batch, 1, seq_len, max_seq_len)  # [SB4P]

    # Fill mask with -inf outside the processed tokens
    attn_mask[:, :, :, current_pos + 1 :] = torch.finfo(attn_mask.dtype).min

    attn_mask = ttnn.from_torch(
        attn_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return xs_1SBH, attn_mask


def prepare_rotation_mat_ttnn(head_dim, max_seq_len, ext_scale_factor, mesh_device):
    """
    Prepare rotation matricies for decode mode.
    """
    rot_mat = get_rotation_mat(dhead=head_dim, end=max_seq_len * 2, ext_scale_factor=ext_scale_factor)
    rot_mats = [
        ttnn.from_torch(
            rot_mat_i.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        for rot_mat_i in rot_mat
    ]

    return rot_mats
