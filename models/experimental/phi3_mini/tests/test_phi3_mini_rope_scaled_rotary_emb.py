# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import AutoModelForCausalLM

from loguru import logger
from models.experimental.phi3_mini.tt.phi3_mini_rope_scaled_rotary_emb import TtPhi3MiniLongRoPEScaledRotaryEmbedding

from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)


def test_phi3_mini_rope_scaled_roatry_emb(device=None):
    if device == None:
        device = ttnn.GetDefaultDevice()
    torch.manual_seed(0)

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
    sequence_size = 384
    torch_position_ids = torch.zeros((1, sequence_size), dtype=torch.bfloat16)
    torch_value_states = torch.randint((1, sequence_size))
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

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaAttention Passed!")
    else:
        logger.warning("RobertaAttention Failed!")

    assert does_pass
