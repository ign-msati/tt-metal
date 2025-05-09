# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.utility_functions import comp_allclose
from transformers import AutoModelForCausalLM
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_phi3_mini_attention_inference(batch: int = 1, seq_len: int = 128, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)
    expected_pcc_score = 0.1

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers.{SELF_ATTN_LAYER_INDEX}.self_attn"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini attn layer
    torch_model = model.model.layers[SELF_ATTN_LAYER_INDEX].self_attn

    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniAttention(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
        layer_idx=SELF_ATTN_LAYER_INDEX,
    )

    # Run torch model
    torch_hidden_states = torch.rand(batch, seq_len, 3072)
    # torch_hidden_states = torch.rand(1, batch, seq_len)
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    torch_output = torch_model(torch_hidden_states, position_ids=torch_position_ids)

    # Run tt model
    tt_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_postion_ids = ttnn.from_torch(
        torch_position_ids[:, None, :].float(),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(tt_hidden_states, position_ids=tt_postion_ids)

    does_pass, pcc_message = assert_with_pcc(
        torch_output[0], ttnn.to_torch(tt_output[0][0]).to(torch_output[0].dtype), expected_pcc_score
    )

    logger.info(comp_allclose(torch_output[0], ttnn.to_torch(tt_output[0][0]).to(torch_output[0].dtype)))
    if does_pass:
        logger.success(f"Phi-3-mini Attention Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini Attention Failed! --> PCC: {pcc_message}")

    # Close device
    ttnn.close_device(device)
