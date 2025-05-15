import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.utility_functions import comp_pcc, comp_allclose
from transformers import AutoModelForCausalLM
from ttnn import ConcatMeshToTensor
from models.experimental.grok.tt.grok_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn
from models.experimental.phi3_mini.tt.model_config import TtPhi3MiniKernelConfigs


def test_phi3_mini_attention_inference(batch: int = 1, seq_len: int = 128, fall_back_to_torch=False, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    expected_pcc_score = 0.1
    dtype = ttnn.bfloat8_b
    batch = 32
    seq_len = 1  # length to generate
    generation_start_pos = 0  # Ref model can only start from pos 0
    generation_length = 1
    ref_past_key_value = None
    all_tests_pass = True

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers.{SELF_ATTN_LAYER_INDEX}.self_attn"
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    state_dict = base_model.state_dict()
    model_config = base_model.config

    # Torch phi3-mini attn layer
    reference_model = base_model.model.layers[SELF_ATTN_LAYER_INDEX].self_attn

    # Tt phi3-mini attn layer
    kernel_args = TtPhi3MiniKernelConfigs(device=device)
    tt_model = TtPhi3MiniAttention(
        config=model_config,
        state_dict=state_dict,
        base_address=base_address,
        layer_idx=SELF_ATTN_LAYER_INDEX,
        device=device,
        kernel_args=kernel_args,
    )

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    max_seq_len = model_config.max_position_embeddings // 128
    rot_mat = prepare_rotation_mat_ttnn(
        head_dim,
        max_seq_len,
        tt_model.mesh_device,
    )

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_config.hidden_size) * 2) - 1
        tt_attention_input = pt_attention_input
        current_pos = generation_start_pos + i

        attention_input, attn_mask = prepare_inputs_ttnn(
            tt_attention_input,
            model_config.hidden_size,
            current_pos,
            tt_model.mesh_device,
        )

        tt_out = tt_model(
            attention_input,
            rot_mat,
            current_pos,
            attn_mask,
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del attention_input, attn_mask
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0))[0].squeeze(2).view(batch, 1, -1)
        )  # [ batch, seq, hidden_dim]

        positions = torch.LongTensor([current_pos])
        reference_output, _, ref_past_key_value = reference_model(
            pt_attention_input,
            past_key_value=ref_past_key_value,
            position_ids=positions,
            use_cache=True,
        )

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, expected_pcc_score)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[current_pos={current_pos}] Phi3_Mini_Attention Passed!")
        else:
            logger.warning(f"[current_pos={current_pos}] Phi3_Mini_Attention Failed!")
            all_tests_pass = False
    if all_tests_pass:
        logger.info("Phi3_Mini Attention output Passed!")
    else:
        logger.warning("Phi3_Mini Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {expected_pcc_score} for some of the outputs. Check Warnings!"

    # Close device
    ttnn.close_device(device)
