import torch
import ttnn
import time

from loguru import logger

from models.experimental.phi3_mini_may_ver_2.tt.phi3_mini_attention import TtPhi3MiniAttention
from models.utility_functions import comp_pcc, comp_allclose
from transformers import AutoModelForCausalLM, DynamicCache
from ttnn import ConcatMeshToTensor
from models.experimental.phi3_mini_may_ver_2.tt.model_config import TtPhi3MiniKernelConfigs
from models.experimental.phi3_mini_may_ver_2.tt.phi3_mini_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn


def test_phi3_mini_attention_inference(
    batch: int = 32, seq_len: int = 1, generation_length: int = 10, attn_mode: str = "decode", device=None
):
    if device == None:
        device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    expected_pcc_score = 0.99
    dtype = ttnn.bfloat8_b
    generation_start_pos = 0  # Ref model can only start from pos 0
    if attn_mode == "prefill":
        num_iters = batch
    else:
        num_iters = generation_length
        seq_len = 1

    ref_past_key_value = DynamicCache()
    all_tests_pass = True

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"model.layers.{SELF_ATTN_LAYER_INDEX}.self_attn"
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    state_dict = base_model.state_dict()
    model_config = base_model.config

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    max_seq_len = model_config.max_position_embeddings // 16

    # Torch phi3-mini attn layer
    reference_model = base_model.model.layers[SELF_ATTN_LAYER_INDEX].self_attn
    transformation_mats = None

    # Setup RoPE transformation matrices
    long_factor = torch.tensor(model_config.rope_scaling["long_factor"], dtype=torch.float32)
    short_factor = torch.tensor(model_config.rope_scaling["short_factor"], dtype=torch.float32)

    # Tt phi3-mini attn layer
    kernel_args = TtPhi3MiniKernelConfigs(device=device)
    tt_model = TtPhi3MiniAttention(
        config=model_config,
        state_dict=state_dict,
        base_address=base_address,
        layer_idx=SELF_ATTN_LAYER_INDEX,
        device=device,
        kernel_args=kernel_args,
        max_batch_size=batch,
        transformation_mats=transformation_mats,
    )

    long_factor_rot_mat = prepare_rotation_mat_ttnn(
        head_dim,
        max_seq_len,
        ext_scale_factor=long_factor,
        mesh_device=tt_model.mesh_device,
        attn_mode=attn_mode,
    )
    short_factor_rot_mat = prepare_rotation_mat_ttnn(
        head_dim,
        max_seq_len,
        ext_scale_factor=short_factor,
        mesh_device=tt_model.mesh_device,
        attn_mode=attn_mode,
    )
    rot_mats = [long_factor_rot_mat, short_factor_rot_mat]

    total_time = 0.0
    pcc_cum = ""
    for i in range(num_iters):
        current_pos = generation_start_pos + i

        if attn_mode == "prefill":
            pt_attention_input = torch.rand(1, seq_len, model_config.hidden_size)
            if seq_len > model_config.original_max_position_embeddings:
                current_iter_rot_mats = rot_mats[0][:, :seq_len, :, :]
            else:
                current_iter_rot_mats = rot_mats[1][:, :seq_len, :, :]
        else:
            pt_attention_input = torch.rand(batch, 1, model_config.hidden_size)
            if current_pos > model_config.original_max_position_embeddings:
                current_iter_rot_mats = rot_mats[0][current_pos]
            else:
                current_iter_rot_mats = rot_mats[1][current_pos]
            # position_ids = torch.Tensor([current_pos]*8)
            # rot_mats = [long_rope_setup.get_rot_mats(position_ids), short_rope_setup.get_rot_mats(position_ids)]
            # if current_pos > self.original_max_seq_len:
            #     current_iter_rot_mats = rot_mats[0]
            #     transformation_mats_decode = self.transformation_mats[0][0]
            # else:
            #     current_iter_rot_mats = rot_mats[1]
            #     transformation_mats_decode = self.transformation_mats[1][0]

        attention_input, _, torch_attn_mask = prepare_inputs_ttnn(
            pt_attention_input,
            model_config.hidden_size,
            current_pos,
            tt_model.mesh_device,
            max_seq_len,
            attn_mode=attn_mode,
        )

        start_time = time.perf_counter()
        tt_out = tt_model(
            attention_input,
            current_iter_rot_mats,
            current_pos,
            None,
            mode=attn_mode,
        )
        if i > 0:
            total_time += time.perf_counter() - start_time

        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        if attn_mode == "prefill":
            tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0))[
                0
            ]  # [ batch, seq, hidden_dim]
        else:
            tt_output_torch = (
                ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0))[0].squeeze(2).view(batch, 1, -1)
            )  # [ batch, seq, hidden_dim]

        if attn_mode == "prefill":
            positions = torch.arange(0, seq_len, 1, dtype=torch.long).unsqueeze(0)
            ref_past_key_value = DynamicCache()
            reference_output, _, _ = reference_model(
                pt_attention_input,
                past_key_value=ref_past_key_value,
                position_ids=positions,
                attention_mask=torch_attn_mask,
            )
        else:
            positions = torch.LongTensor([[current_pos]] * batch)
            reference_output, _, ref_past_key_value = reference_model(
                pt_attention_input,
                past_key_value=ref_past_key_value,
                position_ids=positions,
            )
        del attention_input, _, torch_attn_mask

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, expected_pcc_score)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        pcc_cum = pcc_cum + f"current_pos={current_pos} pcc: {pcc_message}\n"
        if passing:
            logger.info(f"[current_pos={current_pos}] Phi3_Mini_Attention Passed!")
        else:
            logger.warning(f"[current_pos={current_pos}] Phi3_Mini_Attention Failed!")
            all_tests_pass = False

    logger.info(f"PCC per iter: {pcc_cum}\n")
    logger.info(f"Time 1-N iters: {total_time}")
    if all_tests_pass:
        logger.info("Phi3_Mini Attention output Passed!")
    else:
        logger.warning("Phi3_Mini Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {expected_pcc_score} for some of the outputs. Check Warnings!"

    # Close device
    ttnn.close_device(device)
