# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.common import (
    sample_host,
    PagedAttentionConfig,
)
# from models.tt_transformers.tt.model_config import ModelArgs, DecodersPrecision
# from models.tt_transformers.tt.model_config import ModelArgs, DecodersPrecision
from models.tt_transformers.tt.model_config import  DecodersPrecision
from models.tt_transformers.tt.model_config import ModelArgs
# from models.experimental.phi3_mini_may_ver_5.tt.model_config import ModelArgs
# from models.tt_transformers.tt.model import Transformer
from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_model import Phi3Transformer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from transformers import AutoModelForCausalLM, StaticCache, DynamicCache
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from transformers import AutoModelForCausalLM, DynamicCache
from time import time
@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Failing on DRAM harvested P100a, see #21419")
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights, layers",
    [
        # ("random", 1),
        ("instruct", None),
    ],
    ids=[
        # "quick", 
         "full"
         ],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
    # (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    # (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
    # (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
    # (1024*32,),  # For decode-only unit test, there's no need to run with large sequence lengths
    # (1024*64,),  # For decode-only unit test, there's no need to run with large sequence lengths
    # ((1024*128) + 200,),  # For decode-only unit test, there's no need to run with large sequence lengths
    (1024*64,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        # lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=[
        "performance",
        #   "accuracy"
          ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_model_inference(
    weights,
    layers,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_program_cache,
    reset_seeds,
    # ensure_gc,
    request,
):
    ref_past_key_value = DynamicCache()
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = layers == 1  # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    dtype = ttnn.bfloat8_b

    test_id = request.node.callspec.id
    mode_accuracy = "accuracy" in test_id
    # instruct = False  # True if weights == "instruct" else False
    instruct=True
    dummy_weights = True if weights == "random" else False
    dummy_weights=False
    model_args = ModelArgs(
        mesh_device,
        # instruct=instruct,
        dummy_weights=dummy_weights,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    # import pdb; pdb.set_trace()

    # Define minimum PCC for each iteration
    if layers == 1:
        pcc = 0.88 if mode_accuracy else 0.86
    else:
        pcc = 0.94 if mode_accuracy else 0.86

    if layers == 1:  # quick mode has tight PCC checks for known models

        iterations = 10
    else:
        iterations = 15
    #######################333
    final_model_pcc=0.9
    ##########################
    if layers is not None:
        model_args.n_layers = layers
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    # reference_state_dict = {
    #     k[len(state_dict_prefix) :]: v
    #     for k, v in state_dict.items()
    #     if (
    #         any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
    #         or any(
    #             [
    #                 f"{state_dict_prefix}{name}" in k
    #                 for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
    #             ]
    #         )
    #     )
    # }

    # prompts = ["This is a test"] * model_args.max_batch_size
    prompts = ["Capital of india"] * model_args.max_batch_size
    # if dummy_weights:
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        tokenizer = model_args.tokenizer
        if instruct:
            chat = [{'role': 'user', 'content': prompts[0]}]
            print(f"{chat=}")
            tpl = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            # encoded_prompts = tpl['input_ids'] # [model_args.encode_prompt(prompt) for prompt in prompts]
            encoded_prompts = tpl['input_ids'] # [model_args.encode_prompt(prompt) for prompt in prompts]
        else:
            encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]
    # import pdb; pdb.set_trace()
    # print(f"{instruct=}")
    # print(f"{chat=}")
    # print(f"{encoded_prompts.shape=}")
    if run_ref_pt:
        LAYER_INDEX = 0
        LAYER_INDEX_SPLIT=32

        # base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
        base_model = model_args.reference_ign_model
        # reference_model = base_model.model[1:]
        # torch_model_embedded = base_model.model.embed_tokens
        torch_model_decode = base_model.model.layers[:LAYER_INDEX_SPLIT]
        torch_model_norm = base_model.model.norm
        torch_model_lm_head = base_model.lm_head
        # reference_model = model_args.reference_transformer()
        # reference_model.load_state_dict(reference_state_dict)

    # Embedding on host
    embd = model_args.reference_embedding()
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # generation_start_pos = 64*1024
    # generation_start_pos = 128
    # generation_start_pos = 128*1024
    generation_start_pos = 0
    generation_length = iterations

    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TTNN model
    tt_model = Phi3Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True
        final_tests_pass = True
        kv_cache_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    # Keep track of generated outputs to print out later
    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        print("iterations *************^^^^^^^^^^^^^^^^", i)
        iteration_time_start = time()
        logger.info(f"[Model] Generating token {i}")

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        print("Token per second *************^^^^^^^^^^^^^^^^", tokens_per_second_per_user)
        logger.info(
                # f"Iteration {iteration}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user  ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)
                f"Iteration {i}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user  ({1*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

        # Convert ttnn tensor to torch tensor
        mesh_composer = ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
        )
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        )

        ttnn.deallocate(tt_out)

        if run_ref_pt:  # Run reference model
            # In this test all users have the same position

            # ref_output = reference_model(pt_decode_input, current_pos[0])

            ###############################################################33
             #################################################33
        # torch_output_embedded = torch_model_embedded(sample_input_torch)#,
            positions = torch.LongTensor([[current_pos]] * batch)
            torch_output_states=pt_decode_input
            print(f"{positions.shape=}")
            print(f"{torch_output_states.shape=}")
            for layer in range(LAYER_INDEX_SPLIT):
                # print("layer done********************", layer)
                #############################################33
                torch_output_states = torch_model_decode[layer](torch_output_states, 
                                            position_ids=positions, 
                                            # position_ids=current_pos[0], 
                                            past_key_value=ref_past_key_value,
                                        #  past_key_value=ref_past_key_value[i],
                                            use_cache=True)
                torch_output_states=torch_output_states[0]
            # reference_output=torch_output_states
            #     ###############################################

            torch_output_states = torch_model_norm(torch_output_states)#,
            ref_output = torch_model_lm_head(torch_output_states)#,
        ######################################################
            ###############################################################33

        # Increment position
        # current_pos = torch.tensor([generation_start_pos + i  for _ in range(batch)])
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        # Append the generated token to the list of outputs /prefill
        if i in range(len(encoded_prompts[0])):
            # While in "prefill" mode, use the prompt tokens as the output
            all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            if run_ref_pt:
                # Sample from reference model first
                _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])

                # Use the same token for TT model (teacher forcing)
                tt_decode_input = pt_decode_input
                all_outputs.append(pt_out_tok.squeeze(1).tolist()[0])
            else:
                # If not running reference model, sample from TT model directly
                _, tt_out_tok = sample_host(tt_output_torch, temperature=0, top_p=0.8)
                tt_decode_input = embd(tt_out_tok)
                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])

        # Measure PCC if also running reference model
        if run_ref_pt:
            if layers == 1 and i == iterations - 1:  # On last iteration in the quick test, set a tighter PCC
                passing, pcc_message = comp_pcc(ref_output, tt_output_torch, final_model_pcc)
                if not passing:
                    final_tests_pass = False
            else:
                passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"PCC: {pcc_message}")

            if passing:
                logger.info("Model Passed!")
            else:
                logger.warning("Model Failed!")
            if not passing:
                all_tests_pass = False

            # Compare KV caches
            if cache_pcc:
                for l in range(model_args.n_layers):
                    pytorch_layer_present = [
                        reference_model.layers[l]
                        .attention.cache_k.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        reference_model.layers[l]
                        .attention.cache_v.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                    ]
                    tt_layer_present = []
                    if paged_attention:
                        for layer_past in tt_model.layers[l].attention.layer_past:
                            tt_layer_present.append(
                                ttnn.to_torch(
                                    layer_past,
                                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                                        mesh_device,
                                        dims=(1, 3) if model_args.is_galaxy else (0, 1),
                                        mesh_shape=model_args.cluster_shape,
                                    ),
                                )[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
                                .reshape(
                                    model_args.max_batch_size,
                                    paged_attention_config.max_num_blocks // model_args.max_batch_size,
                                    model_args.n_kv_heads,
                                    paged_attention_config.block_size,
                                    model_args.head_dim,
                                )
                                .transpose(1, 2)
                                .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
                                    :batch, ...
                                ]
                            )
                    else:
                        for layer_past in tt_model.layers[l].attention.layer_past:
                            tt_layer_present.append(
                                ttnn.to_torch(
                                    layer_past,
                                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                                        mesh_device,
                                        dims=(1, 0) if model_args.is_galaxy else (0, 1),
                                        mesh_shape=model_args.cluster_shape,
                                    ),
                                )[:batch, :, :, :]
                            )

                    for kv_cache, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = min(
                            model_args.max_seq_len, generation_start_pos + generation_length + 1
                        )
                        cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                        if (
                            layers == 1 and i == iterations - 1
                        ):  # On last iteration in the quick test, set a tighter PCC
                            if kv_cache == 0:  # K cache
                                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, final_k_cache_pcc)
                            else:  # V cache
                                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, final_v_cache_pcc)
                        else:
                            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                        if kv_cache == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"KV Cache Passed!")
                        else:
                            logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                            all_tests_pass = False

        if not dummy_weights:
            logger.info("[ttnn generation User 0] " + tokenizer.decode(all_outputs).replace("\n", "\\n"))
            if run_ref_pt:
                logger.info("[Ref generation User 0] " + tokenizer.decode(all_outputs_ref).replace("\n", "\\n"))
    # print(f"{chat=}")
    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} decode iterations Passed!")
        else:
            logger.warning("One or more iterations of decode had bad PCC")
            if layers == 1:
                assert final_tests_pass, f"PCC value is lower than {final_model_pcc} for final output. Check Warnings!"
            assert kv_cache_tests_pass, f"KV Cache PCC value is lower expected for some of the outputs. Check Warnings!"
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
