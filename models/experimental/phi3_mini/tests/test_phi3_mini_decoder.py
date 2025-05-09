# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.experimental.phi3_mini.tt.phi3_mini_decoder import TtPhi3MiniDecoder
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def test_phi3_mini_decoder_inference(device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)

    LAYER_INDEX = 0
    base_address = f"model.layers.{LAYER_INDEX}"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini attn layer
    torch_model = model.model.layers[LAYER_INDEX]



    hidden_size =3072
    seq_len=3
    batch=1
    # hidden_states, tt_hidden_states = create_attention_input(mode, ttnn.bfloat16, batch, seq_len, config.hidden_size, ttnn_device)
    torch_hidden_states = torch.rand(batch, seq_len, hidden_size)
    attention_mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool).tril()
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    torch_output = torch_model(torch_hidden_states, position_ids=torch_position_ids)

    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniDecoder(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
        layer_idx=LAYER_INDEX,
    )

    # # Run torch model
    # hidden_states = torch.rand(1, 32, 768)
    # attention_mask = torch.ones(1, 1, 32)
    # torch_output = torch_model(hidden_states, attention_mask=attention_mask)

    # Run tt model
    tt_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    # tt_attention_mask = ttnn.from_torch(
    #     attention_mask,
    #     device=device,
    #     dtype=ttnn.bfloat16,
    #     layout=ttnn.TILE_LAYOUT,
    # )
    tt_output = tt_model(tt_hidden_states, position_ids=torch_position_ids)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("phi3decoder Passed!")
    else:
        logger.warning("phi3decoder Failed!")

    assert does_pass
    # Close device
    ttnn.close_device(device)

test_phi3_mini_decoder_inference()