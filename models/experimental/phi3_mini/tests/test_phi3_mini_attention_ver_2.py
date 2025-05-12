# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.phi3_mini.tt.phi3_mini_attention_ver_2 import TtPhi3MiniAttention
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
)
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import sys
from models.experimental.phi3_mini.tt.dump_data import dump_np_array
import time

def test_phi3_mini_attention_inference(fall_back_to_torch=True):  # device):
    torch.manual_seed(1234)
    ttnn_device = ttnn.open_device(device_id=0)
    ###########################################################333
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    # Torch phi3-mini mlp layer
    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers.{SELF_ATTN_LAYER_INDEX}.mlp"
    model = model.model.layers[SELF_ATTN_LAYER_INDEX].self_attn
    #############################################################
    # base_address = f"model.layers[{SELF_ATTN_LAYER_INDEX}].self_attn"

    # model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # # Torch phi3-mini attn layer
    # torch_model = model.model.layer[SELF_ATTN_LAYER_INDEX].self_attn

    # Run torch model
    hidden_size = 3072
    seq_len = 3
    batch = 1
    # hidden_states, tt_hidden_states = create_attention_input(mode, ttnn.bfloat16, batch, seq_len, config.hidden_size, ttnn_device)
    hidden_states = torch.rand(batch, seq_len, hidden_size)
    attention_mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool).tril()
    # torch_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    torch_position_ids = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0).repeat(batch, 1).float()

    torch_past_key_values = DynamicCache()

    # Attention prefill
    torch_output, torch_attention_weights, torch_past_key_values = model(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=torch_position_ids,
        # past_key_value=torch_past_key_values,
        # use_cache=True,
    )

    #################################3333333
    # TTNN attention
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, device=ttnn_device, prefix=f"model.layers.self_attn"
    )
    # Tt phi3-mini attn layer
    tt_model = TtPhi3MiniAttention(
        device=ttnn_device,
        config=model.config,
        parameters=parameters,
        # base_address=base_address,
        # state_dict=model.state_dict(),
        layer_idx=SELF_ATTN_LAYER_INDEX,
    )

    # tt_position_ids = ttnn.from_torch(
    #     torch_position_ids,
    #     dtype=ttnn.bfloat16,
    #     device=ttnn_device,
    #     layout=ttnn.TILE_LAYOUT,
    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # )

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if fall_back_to_torch:
        tt_postion_ids = torch_position_ids
    else:
        tt_postion_ids = ttnn.from_torch(
            torch_position_ids,
            device=ttnn.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
    # tt_position_ids = ttnn.from_torch(
    #     torch_position_ids,
    #     dtype=ttnn.bfloat16,
    #     device=ttnn_device,
    #     layout=ttnn.TILE_LAYOUT,
    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # )

    # TTNN attention prefill
    start = time.time()
    tt_output, tt_layer_present = tt_model(
        tt_hidden_states,
        attention_mask=tt_attention_mask,
        position_ids=tt_postion_ids,
        # position_ids=torch_position_ids,
        past_key_values=None,
        use_cache=True,
        fall_back_to_torch=fall_back_to_torch
    )
    end = time.time()
    duration = end - start
    print(f"total duration {duration}")
    # # Compare outputs
    tt_output_torch = ttnn.to_torch(tt_output).to(torch_output.dtype)
    tt_output_torch = tt_output_torch.squeeze(0)

    print("torch_output", torch_output)
    dump_np_array(torch_output, "expected.txt")
    dump_np_array(tt_output_torch, "result.txt")
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("phi3mini Passed!")
    else:
        logger.warning("phi3mini Failed!")

    assert does_pass
    ttnn.close_device(ttnn_device)


test_phi3_mini_attention_inference()