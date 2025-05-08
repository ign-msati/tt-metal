# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import AutoModelForCausalLM

from loguru import logger
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding

from models.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_phi3_mini_rope_scaled_roatry_emb(batches: int = 1, seq_len: int = 384, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    expected_pcc_score = 0.97

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini RoPE layer
    torch_model = model.model.layers[0].self_attn.rotary_emb

    rotary_dim = model.config.hidden_size // model.config.num_attention_heads
    # Tt phi3-mini mlp layer
    tt_model = TtPhi3MiniLongRoPEScaledRotaryEmbedding(
        dim=rotary_dim,
        config=model.config,
        device=device,
    )

    # Run torch model
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0)
    torch_value_states = torch.arange(0, seq_len, 1, dtype=torch.long)
    torch_output = torch_model(torch_value_states, torch_position_ids)

    # Run tt model
    tt_postion_ids = ttnn.from_torch(
        torch_position_ids[:, None, :].float(),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_value_states = ttnn.from_torch(
        torch_value_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_value_states, tt_postion_ids, seq_len)

    does_pass_cos, pcc_message_cos = assert_with_pcc(
        torch_output[0], ttnn.to_torch(tt_output[0]).to(torch_output[0].dtype), expected_pcc_score
    )
    does_pass_sin, pcc_message_sin = assert_with_pcc(
        torch_output[1], ttnn.to_torch(tt_output[1]).to(torch_output[1].dtype), expected_pcc_score
    )

    logger.info(comp_allclose(torch_output[0], ttnn.to_torch(tt_output[0]).to(torch_output[0].dtype)))
    logger.info(comp_allclose(torch_output[1], ttnn.to_torch(tt_output[1]).to(torch_output[1].dtype)))
    if does_pass_cos and does_pass_sin:
        logger.success(
            f"Phi-3-mini RoPE Scaled Rotary Embedding Passed! --> PCC_COS: {pcc_message_cos}, PCC_SIN: {pcc_message_sin}"
        )
    else:
        logger.warning(
            f"Phi-3-mini RoPE Scaled Rotary Embedding Failed! --> PCC_COS: {pcc_message_cos}, PCC_SIN: {pcc_message_sin}"
        )

    # Close device
    ttnn.close_device(device)
