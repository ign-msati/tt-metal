import torch
import ttnn
from ttnn import ReplicateTensorToMesh


def precompute_freqs(
    dim: int,
    end: int,
    ext_scale_tensor: torch.Tensor,
    scale_factor=None,
    theta: float = 10000.0,
    convert_to_tt_tensor: bool = False,
    mesh_device=None,
):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions, grok-style.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Grok-1 uses 10000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (ext_scale_tensor * theta ** (torch.arange(0, dim, 2).float() / dim))
    if scale_factor is None:
        scale_factor = 1.1902380714238083  # TODO: calculate from model config instead of hardcoding
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.cos(emb) * scale_factor, torch.sin(emb) * scale_factor

    if convert_to_tt_tensor:
        cos = ttnn.from_torch(
            cos.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        sin = ttnn.from_torch(
            sin.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        # cos = ttnn.reshape(
        #     cos, cos.shape, (1, 1, 128, 128)
        # )
        # sin = ttnn.reshape(
        #     sin, sin.shape, (1, 1, 128, 128)
        # )
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


def get_rotation_mat(dhead, end, ext_scale_factor, attn_mode):
    cos, sin = precompute_freqs(dhead, end, ext_scale_factor)
    if attn_mode == "prefill":
        rot_mat = torch.stack([freq_row_to_rotation_matrix(c, s) for c, s in zip(cos, sin)], dim=0)
    else:
        rot_mat = [freq_row_to_rotation_matrix(c, s) for c, s in zip(cos, sin)]
    return rot_mat


def prepare_inputs_ttnn(x_bsh, hidden_size, current_pos, mesh_device, max_seq_len, attn_mode):
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

    if attn_mode == "prefill":
        assert batch == 1
        x_1BSH = x_bsh.unsqueeze(0)
        input = x_1BSH

        # Attention mask
        attn_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len) * float("-inf"), diagonal=1)

    elif attn_mode == "decode":
        assert seq_len == 1
        x_1SBH = x_bsh.view(1, seq_len, batch, hidden_size)
        input = x_1SBH

        # Attention mask
        attn_mask = torch.zeros(batch, 1, 1, max_seq_len)
        # Fill mask with -inf outside the processed tokens
        attn_mask[:, :, :, current_pos + 1 :] = torch.finfo(attn_mask.dtype).min

    # input goes to L1
    input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_attn_mask = ttnn.from_torch(
        attn_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input, tt_attn_mask, attn_mask


def prepare_rotation_mat_ttnn(head_dim, max_seq_len, ext_scale_factor, mesh_device, attn_mode):
    """
    Prepare rotation matricies for decode mode.
    """
    rot_mat = get_rotation_mat(
        dhead=head_dim, end=max_seq_len * 2, ext_scale_factor=ext_scale_factor, attn_mode=attn_mode
    )

    if attn_mode == "prefill":
        rot_mats = ttnn.from_torch(
            rot_mat.unsqueeze(0),  # 1,max_seq_len,head_dim,head_dim
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
    else:
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
