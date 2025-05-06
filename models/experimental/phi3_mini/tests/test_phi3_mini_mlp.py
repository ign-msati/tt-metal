# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import AutoModelForCausalLM

from loguru import logger
from models.experimental.phi3_mini.tt.phi3_mini_mlp import TtPhi3MiniMLP

from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)


def test_phi3_mini_mlp(device=None):
    if device == None:
        device = ttnn.GetDefaultDevice()
    torch.manual_seed(1234)

    SELF_MLP_LAYER_INDEX = 0
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    base_address = f"model.layers.{SELF_MLP_LAYER_INDEX}.mlp"

    # Torch phi3-mini mlp layer
    torch_model = model.model.layers[SELF_MLP_LAYER_INDEX].mlp

    # Tt phi3-mini mlp layer
    tt_model = TtPhi3MiniMLP(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    # Run torch model
    hidden_states = torch.rand(1, 43, 768)
    torch_output = torch_model(hidden_states)

    # Run tt model
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_hidden_states)

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
