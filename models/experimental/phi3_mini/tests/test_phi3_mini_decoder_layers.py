# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_decoder_layer import TtPhi3MiniDecoderLayer
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
def test_phi3_mini_Decode_layer_inference(device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)


    hidden_size =3072
    seq_len=3
    batch=1
    fall_back_to_torch=False
    LAYER_INDEX_SPLIT = 32
    # base_address = f"model.layers.{LAYER_INDEX}"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini attn layer
    torch_model = model.model.layers[:LAYER_INDEX_SPLIT]
    # torch_model = model

    # hidden_states, tt_hidden_states = create_attention_input(mode, ttnn.bfloat16, batch, seq_len, config.hidden_size, ttnn_device)
    torch_hidden_states = torch.rand(batch, seq_len, hidden_size)
    attention_mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool).tril()
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    torch_output_states=torch_hidden_states
    for i in range(LAYER_INDEX_SPLIT):
        torch_output_states = torch_model[i](torch_output_states, position_ids=torch_position_ids)
        torch_output_states=torch_output_states[0]

    torch_output=torch_output_states
    # print("torch ouput", torch_output)
    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniDecoderLayer(
        config=model.config,
        device=device,
        state_dict=model.state_dict(),
        LAYER_INDEX_SPLIT=LAYER_INDEX_SPLIT,
    )

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
    if fall_back_to_torch:
        tt_postion_ids = torch_position_ids
    else:
        tt_postion_ids = ttnn.from_torch(
            torch_position_ids,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
    tt_output = tt_model(tt_hidden_states, position_ids=tt_postion_ids)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)


    if does_pass:
        logger.success(f"Phi-3-mini Decoder layer Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini Decoder layers Failed! --> PCC: {pcc_message}")


    assert does_pass
    # Close device
    ttnn.close_device(device)


test_phi3_mini_Decode_layer_inference()