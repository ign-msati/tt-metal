# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def test_phi3_mini_attention_inference(device):
    torch.manual_seed(1234)

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers[{SELF_ATTN_LAYER_INDEX}].self_attn"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Torch phi3-mini attn layer
    torch_model = model.model.layer[SELF_ATTN_LAYER_INDEX].self_attn

    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniAttention(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
        layer_idx=SELF_ATTN_LAYER_INDEX,
    )

    # Run torch model
    hidden_states = torch.rand(1, 32, 768)
    attention_mask = torch.ones(1, 1, 32)
    torch_output = torch_model(hidden_states, attention_mask=attention_mask)

    # Run tt model
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_hidden_states, attention_mask=tt_attention_mask)

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
