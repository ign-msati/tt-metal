import torch
import ttnn

from loguru import logger

from models.experimental.phi3_mini_may.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.utility_functions import comp_pcc, comp_allclose
from transformers import AutoModelForCausalLM, DynamicCache
from ttnn import ConcatMeshToTensor
from models.experimental.phi3_mini_may.tt.model_config import TtPhi3MiniKernelConfigs
from models.experimental.phi3_mini_may.tt.phi3_mini_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn


def test_phi3_mini_attention_inference(batch: int = 1, seq_len: int = 128, fall_back_to_torch=False, device=None):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    expected_pcc_score = 0.99
    dtype = ttnn.bfloat8_b
    batch = 8
    seq_len = 1  # length to generate
    generation_start_pos = 0  # Ref model can only start from pos 0
    generation_length = 10
    ref_past_key_value = None
    all_tests_pass = True

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers.{SELF_ATTN_LAYER_INDEX}.self_attn"
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    state_dict = base_model.state_dict()
    model_config = base_model.config

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    max_seq_len = model_config.max_position_embeddings // 128
    # ref_past_key_value = StaticCache(config=model_config, max_batch_size=batch, max_cache_len=1)
    ref_past_key_value = DynamicCache()

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

    long_factor = torch.tensor(model_config.rope_scaling["long_factor"], dtype=torch.float32)
    short_factor = torch.tensor(model_config.rope_scaling["short_factor"], dtype=torch.float32)
    long_factor_rot_mat = prepare_rotation_mat_ttnn(
        head_dim,
        max_seq_len,
        ext_scale_factor=long_factor,
        mesh_device=tt_model.mesh_device,
    )
    short_factor_rot_mat = prepare_rotation_mat_ttnn(
        head_dim,
        max_seq_len,
        ext_scale_factor=short_factor,
        mesh_device=tt_model.mesh_device,
    )

    pcc_cum = ""
    for i in range(generation_length):
        pt_attention_input = torch.rand(batch, seq_len, model_config.hidden_size)
        tt_attention_input = pt_attention_input
        current_pos = generation_start_pos + i

        attention_input, attn_mask = prepare_inputs_ttnn(
            tt_attention_input,
            model_config.hidden_size,
            current_pos,
            tt_model.mesh_device,
            max_seq_len,
        )

        tt_out = tt_model(
            attention_input,
            short_factor_rot_mat,
            current_pos,
            attn_mask,
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del attention_input, attn_mask
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0))[0].squeeze(2).view(batch, 1, -1)
        )  # [ batch, seq, hidden_dim]

        positions = torch.LongTensor([[current_pos]] * batch)
        reference_output, _, ref_past_key_value = reference_model(
            pt_attention_input,
            past_key_value=ref_past_key_value,
            position_ids=positions,
        )

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, expected_pcc_score)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        pcc_cum = pcc_cum + f"current_pos={current_pos} pcc: {pcc_message}\n"
        if passing:
            logger.info(f"[current_pos={current_pos}] Phi3_Mini_Attention Passed!")
        else:
            logger.warning(f"[current_pos={current_pos}] Phi3_Mini_Attention Failed!")
            all_tests_pass = False
    if all_tests_pass:
        logger.info("Phi3_Mini Attention output Passed!")
        logger.info(pcc_cum)
    else:
        logger.warning("Phi3_Mini Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {expected_pcc_score} for some of the outputs. Check Warnings!"

    # Close device
    ttnn.close_device(device)
