# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_model_diff_block import TtPhi3MiniModel
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
def test_phi3_mini_model_inference(device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)

    # LAYER_INDEX = 0
    # base_address = f"model.layers.{LAYER_INDEX}"

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    model.eval()
    # Torch phi3-mini attn layer
    # torch_model = model.model.layers[LAYER_INDEX]
    # torch_model = model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")


    hidden_size =3072
    # seq_len=16
    batch=1
    fall_back_to_torch=False
    LAYER_INDEX_SPLIT = 32

    torch_model_embedded = model.model.embed_tokens
    torch_model_decode = model.model.layers[:LAYER_INDEX_SPLIT]
    torch_model_norm = model.model.norm
    torch_model_lm_head = model.lm_head


    # hidden_states, tt_hidden_states = create_attention_input(mode, ttnn.bfloat16, batch, seq_len, config.hidden_size, ttnn_device)
    # promt="how to build the good model"
    promt="find the multiple of 4"
    sample_input_torch = tokenizer(promt,return_tensors="pt")
    seq_len=sample_input_torch["input_ids"].shape[-1]
    sample_input_torch = (sample_input_torch["input_ids"],)[0]
    sample_input_torch=torch.tensor([[  715, 29879,   925,   664]])
    # # sample_input_torch=torch.tensor([[ 76,6783, 715, 29879,   925,   664,757,858,232,343]])
    seq_len=sample_input_torch[0].shape[-1]
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0).repeat(batch, 1)



    torch_output_embedded = torch_model_embedded(sample_input_torch)#,
    torch_output_states=torch_output_embedded
    for i in range(LAYER_INDEX_SPLIT):
        torch_output_states = torch_model_decode[i](torch_output_states, position_ids=torch_position_ids)
        torch_output_states=torch_output_states[0]

    torch_output_states = torch_model_norm(torch_output_states)#,
    torch_output_states = torch_model_lm_head(torch_output_states)#,



    torch_output=torch_output_states

    tt_model = TtPhi3MiniModel(
        config=model.config,
        # base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
        LAYER_INDEX_SPLIT=LAYER_INDEX_SPLIT
    )
    sample_input_tt = ttnn.to_device(ttnn.from_torch(sample_input_torch), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # tt_attention_mask = ttnn.from_torch(
    #     attention_mask,
    #     device=device,
    #     dtype=ttnn.bfloat16,
    #     layout=ttnn.TILE_LAYOUT,
    # )
    # fall_back_to_torch=False
    if fall_back_to_torch:
        tt_postion_ids = torch_position_ids
    else:
        tt_postion_ids = ttnn.from_torch(
            torch_position_ids,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
    tt_output = tt_model(sample_input_tt, position_ids=tt_postion_ids)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)
    # print("shapeeeeeeeeeeeeeee", torch_output[0].shape)
    print("tt_output_torch.shape, torch_output.shape)",tt_output_torch.shape, torch_output[0].shape)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)


    if does_pass:
        logger.success(f"Phi-3-mini Decoder Passed! --> PCC: {pcc_message}")
    else:
        logger.warning(f"Phi-3-mini Decoder Failed! --> PCC: {pcc_message}")


    assert does_pass
    # Close device
    ttnn.close_device(device)

test_phi3_mini_model_inference()