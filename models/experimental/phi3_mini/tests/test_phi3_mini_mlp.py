# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import AutoModelForCausalLM

from loguru import logger
from models.experimental.phi3_mini.tt.phi3_mini_mlp import TtPhi3MiniMLP
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import comp_allclose


def test_phi3_mini_mlp(batch: int = 2, seq_len: int = 3072, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)
    expected_pcc_score = 0.98

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
    hidden_states = torch.rand(1, batch, seq_len)
    torch_output = torch_model(hidden_states)

    # Run tt model
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_hidden_states)

    does_pass, pcc_message = assert_with_pcc(
        torch_output, ttnn.to_torch(tt_output[0]).to(torch_output.dtype), expected_pcc_score
    )

    logger.info(comp_allclose(torch_output, ttnn.to_torch(tt_output[0]).to(torch_output.dtype)))
    if does_pass:
        logger.success(f"Phi-3-mini MLP Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini MLP Failed! --> PCC: {pcc_message}")

    # Close device
    ttnn.close_device(device)
