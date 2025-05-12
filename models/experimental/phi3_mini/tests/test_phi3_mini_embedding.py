# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_embedding import TtPhi3MiniEmbedding
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
def test_phi3_mini_embedding_inference(device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)
    expected_pcc_score = 0.98
    # LAYER_INDEX = 0
    # base_address = f"model.layers.{LAYER_INDEX}"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Torch phi3-mini attn layer
    torch_model = model.model.embed_tokens


    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    promt="even number after 2"
    sample_input_torch = tokenizer(promt,return_tensors="pt")
    sample_input_torch = (sample_input_torch["input_ids"],)[0]
    # sample_input_torch=torch.tensor([[  715, 29879,   925,   664]])
    torch_output = torch_model(sample_input_torch)#, position_ids=torch_position_ids)


    # print("torch ouput", torch_output)
    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniEmbedding(
        config=model.config,
        # base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    # Run tt model
    sample_input_tt = ttnn.from_torch(
        sample_input_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(sample_input_tt)
    ###############################################################
    # Compare outputs
    # tt_output_torch = tt2torch_tensor(tt_output[0])
    # tt_output_torch = tt_output_torch.squeeze(0)

    # does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    # logger.info(comp_allclose(torch_output[0], tt_output_torch))
    # logger.info(pcc_message)
    #######################################3333
    does_pass, pcc_message = assert_with_pcc(
    torch_output, ttnn.to_torch(tt_output).to(torch_output.dtype), expected_pcc_score
    )

    logger.info(comp_allclose(torch_output, ttnn.to_torch(tt_output).to(torch_output.dtype)))
    if does_pass:
        logger.success(f"Phi-3-mini embedding Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini embedding Failed! --> PCC: {pcc_message}")

    # Close device
    ttnn.close_device(device)
    ##############################################


    # if does_pass:
    #     logger.success(f"Phi-3-mini Embeddeding Passed! --> PCC: {pcc_message}")
    # else:
    #     logger.warning(f"Phi-3-mini Embeddeding Failed! --> PCC: {pcc_message}")


    # assert does_pass
    # # Close device
    # ttnn.close_device(device)

test_phi3_mini_embedding_inference()