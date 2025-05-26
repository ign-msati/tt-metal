# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.model_config import ModelArgs
# from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
# from models.experimental.phi3_mini.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_common import get_prefill_rot_mat, get_rot_transformation_mat
from transformers import AutoModelForCausalLM, DynamicCache


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (
        256,  # 4096,
        # 1024 * 32,
        # 1024 * 64,
    ),
)
def test_attention_inference(
    max_seq_len,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    # ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # reference_model = model_args.reference_attention()
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    reference_model = base_model.model.layers[0].self_attn

    # pre-compute the rotational embedding matrix and send to device

    rot_mats = get_prefill_rot_mat(model_args.head_dim, mesh_device, max_seq_len, model_args.rope_ext_scaling, model_args.orig_context_len, start_pos=0)
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)

    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_model = Attention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )

    pt_attention_input = torch.rand(batch_size, max_seq_len, model_args.dim)
    tt_attention_input = pt_attention_input.clone()
    attention_input = model_args.prepare_residual_tensor_prefill(
        tt_attention_input,
        force_replicated=False if model_args.is_galaxy else True,
    )

    tt_out = tt_model(
        attention_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    tt_out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape)
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, max_seq_len, -1)  # [ batch, seq, hidden_dim]

    position_ids = torch.arange(0, max_seq_len, 1, dtype=torch.long).unsqueeze(0)
    torch_attn_mask = torch.triu(torch.ones(1, 1, max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
    ref_past_key_value = DynamicCache()
    reference_output, _, _ = reference_model(
        pt_attention_input,
        past_key_value=ref_past_key_value,
        position_ids=position_ids,
        attention_mask=torch_attn_mask,
    )

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Attention Passed!")
    else:
        logger.warning(f"Attention Failed!")
        all_tests_pass = False
    assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"