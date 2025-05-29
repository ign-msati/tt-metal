# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

from models.tt_transformers.tt.generator import Generator
from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_common import get_max_prefill_chunk_size
from models.tt_transformers.tt.common import (
    get_padded_prefill_len,
    num_blocks_in_seq,
    get_block_size,
)


class Phi3MiniGenerator(Generator):
    def __init__(self, model, model_args, mesh_device, tokenizer=None, formatter=None):
        super().__init__(model, model_args, mesh_device, tokenizer, formatter)

    def prefill_forward_text(self, tokens: torch.Tensor, page_table=None, kv_cache=None, prompt_lens=None, max_generated_tokens=200):
        force_long_context_scaling=False
        batch, batch_seq_len = tokens.shape

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        # Checking if any of the batches cross the original_context_len of the model during decode phase
        # If any of the batches shows this behaviour we force all batches to adopt higher context rope scaling
        # This prevent issues due to dynamic scale switching during decode
        for batch_id, prompt_len in enumerate(prompt_lens):
            batch_context_len = prompt_len + max_generated_tokens
            if (prompt_len <= self.model_args[batch_id].orig_context_len) and (batch_context_len > self.model_args[batch_id].orig_context_len):
                logger.info(f"Found Batch: {batch_id} generation crossing model's original context length of {self.model_args[batch_id].orig_context_len}")
                force_long_context_scaling=True
                logger.info(f"Forcing long context scaling for Model: {self.model_args[batch_id].model_name}")

        data_parallel = min(batch, self.data_parallel)
        batch_per_device = batch // data_parallel

        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
            page_table = torch.chunk(page_table, self.data_parallel, 0)

        out_list = []
        for group_user_id in range(batch_per_device):
            for model_id in range(data_parallel):
                user_id = group_user_id + model_id * batch_per_device

                logger.info(f"Prefilling User {user_id + 1}")
                seq_len = prompt_lens[user_id]
                last_token_idx = seq_len - 1

                prefill_seq_len = get_padded_prefill_len(seq_len)
                prefill_ids = torch.cat(
                    [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
                )
                if page_table is not None:
                    page_table_user = self._get_prefill_user_page_table(
                        page_table[model_id], kv_cache[model_id], seq_len
                    )

                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user if page_table is not None else None,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=kv_cache[model_id] if kv_cache is not None else None,
                    model_id=model_id,
                    force_long_context_scaling=force_long_context_scaling,
                )
                out_list.append(logits)

        # We gather data back to how at the end of prefill
        for idx, out in enumerate(out_list):
            model_id = idx % self.data_parallel
            group_user_id = idx // self.data_parallel
            user_id = group_user_id + model_id * batch_per_device

            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = self.model[model_id].process_output_prefill(
                out, last_token_idx=(last_token_idx % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_single_user_text(self, tokens, page_table, user_id, last_token_idx, kv_cache=None, model_id=-1, force_long_context_scaling=False):
        seq_len = tokens.shape[-1]

        # Context length used for the batch
        batch_prefill_seq_len = last_token_idx + 1

        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        if use_chunked_prefill:
            """
            Chunked prefill requires paged attention. There are some strange constraints which we must meet:
             - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
             checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
             - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
             - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
             to keep it otherwise unaware that it is operating on a chunk.
             - due to the above point, we must always set user_id to 0 for chunked prefill.
            """
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert (
                last_token_idx is not None and last_token_idx < seq_len
            ), "last_token_idx must be provided and less than seq_len"
            min_prefill_chunk_size = self.model_args[model_id].min_prefill_chunk_size if (hasattr(self.model_args[model_id], 'min_prefill_chunk_size')) else None
            chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size, min_prefill_chunk_size)
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                assert (
                    chunk_end <= seq_len
                ), f"Chunk end should be less than seq_len, got chunk_end={chunk_end} and seq_len={seq_len}"
                chunk_tokens = tokens[:, chunk_start:chunk_end]
                chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    batch_prefill_seq_len,
                    force_long_context_scaling=force_long_context_scaling,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats=chunk_rot_mats_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            prefill_input, rot_mats_prefill, page_table_tt, _ = self.model[model_id].prepare_inputs_prefill(
                tokens,
                batch_prefill_seq_len,
                force_long_context_scaling=force_long_context_scaling,
                page_table=page_table,
            )

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats=rot_mats_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )
            return tt_logits
