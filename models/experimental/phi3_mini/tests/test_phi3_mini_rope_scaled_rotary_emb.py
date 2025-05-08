# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import AutoModelForCausalLM

from loguru import logger
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding

from models.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_phi3_mini_rope_scaled_roatry_emb(seq_len: int = 384, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    expected_pcc_score = 0.98

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini RoPE layer
    torch_model = model.model.layers[0].self_attn.rotary_emb

    # Tt phi3-mini mlp layer
    tt_model = TtPhi3MiniLongRoPEScaledRotaryEmbedding(
        dim=None,
        config=model.config,
        device=device,
    )

    # Run torch model
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long)
    torch_value_states = torch.randint((1, seq_len))
    torch_output = torch_model(torch_value_states, torch_position_ids)

    # Run tt model
    tt_postion_ids = ttnn.from_torch(
        torch_position_ids,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_value_states = ttnn.from_torch(
        torch_value_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_value_states, tt_postion_ids)

    does_pass, pcc_message = assert_with_pcc(
        torch_output, ttnn.to_torch(tt_output[0]).to(torch_output.dtype), expected_pcc_score
    )

    logger.info(comp_allclose(torch_output, ttnn.to_torch(tt_output[0]).to(torch_output.dtype)))
    if does_pass:
        logger.success(f"Phi-3-mini RoPE Scaled Rotary Embedding Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini RoPE Scaled Rotary Embedding Failed! --> PCC: {pcc_message}")

    # Close device
    ttnn.close_device(device)
