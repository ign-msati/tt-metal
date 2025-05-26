import torch
import ttnn
from loguru import logger
import math
from models.tt_transformers.tt.common import (
    gather_cos_sin,
    PagedAttentionConfig
)


def precompute_freqs(dim: int, end: int, ext_scale_tensor: torch.Tensor, scale_factor=None, theta: float = 10000.0, convert_to_tt_tensor: bool=False, mesh_device=None):
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
    # freqs = F.pad(freqs, (0,16), "constant", 0)
    if scale_factor is None:
        scale_factor = 1.1902380714238083  # TODO: calculate from model config instead of hardcoding
    t = torch.arange(end)
    # freqs = torch.einsum("i,j->ij", t, freqs)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs) * scale_factor, torch.sin(freqs) * scale_factor


def get_prefill_rot_mat(head_dim, mesh_device, seq_len, rope_scaling_factor, orig_context_len, start_pos=0):
    if seq_len > orig_context_len:
        scaling_factor = torch.tensor(rope_scaling_factor["long_factor"], dtype=torch.float32)
    else:
        scaling_factor = torch.tensor(rope_scaling_factor["short_factor"], dtype=torch.float32)
    cos, sin = precompute_freqs(head_dim, seq_len, scaling_factor)
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_torch(
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_gathereds = ttnn.from_torch(
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    # from models.tt_transformers.tt.model import Transformer
    # from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_model import Transformer
    from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_model import Phi3Transformer
    from models.tt_transformers.tt.model_config import ModelArgs

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    model = Phi3Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict