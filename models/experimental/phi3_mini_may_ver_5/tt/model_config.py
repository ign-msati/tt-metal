# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import math
import torch

from loguru import logger
from models.tt_transformers.tt.model_config import ModelArgs
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import nearest_multiple


class Phi3MiniModelArgs(ModelArgs):
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
    ):
        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )

    def _set_params_from_dict(self, params):
        # Common params with different names between Meta and HF
        self.dim = params.get("hidden_size")
        self.n_heads = params.get("num_attention_heads")
        self.n_kv_heads = params.get("num_key_value_heads")
        self.n_layers = params.get("num_hidden_layers")
        self.full_model_n_layers = self.n_layers
        self.norm_eps = params.get("rms_norm_eps")
        self.vocab_size = params["vocab_size"]
        self.padded_vocab_size = 32 * 1024
        self.head_dim = self.dim // self.n_heads

        # Handle different MLP dimension specifications
        self.hidden_dim = params["intermediate_size"]
        self.ffn_dim_multiplier = None
        self.multiple_of = None

        if "_name_or_path" in params:
            self.model_name = os.path.basename(params["_name_or_path"])

        self.unpadded_hidden_dim = self.hidden_dim
        # Don't need to pad for CPU runs
        if self.num_devices:
            # Default padding cores for each model, 0 if not set here
            default_padded_cores = 0

            # Override MLP padding cores from env var
            mlp_padded_cores = int(os.environ.get("PAD_MLP_CORES", default_padded_cores))

            # Only pad if MLP_PADDED_CORES is non-zero
            if mlp_padded_cores > 0:
                padded_hidden_dim = nearest_multiple(
                    self.hidden_dim, mlp_padded_cores * self.tile_size * self.num_devices
                )
                if padded_hidden_dim != self.hidden_dim:
                    logger.info(
                        f"PAD_MLP_CORES={mlp_padded_cores}, padding hidden dim from {self.hidden_dim} to {padded_hidden_dim}"
                    )
                    self.hidden_dim = padded_hidden_dim

        # RoPE params
        self.rope_theta = params.get("rope_theta")
        # If use_scaled_rope is not present, assume setting rope_scaling means use scaled rope
        # If it is present and is set to false, do not use scaled rope
        # Setting self.rope_scaling_factor to None is our way of saying do not use scaled rope
        self.rope_scaling_factor = None
        if "rope_scaling" in params:
            self.max_context_len = params.get("max_position_embeddings", None)
            self.orig_context_len = params.get("original_max_position_embeddings", None)
            self.rope_scaling = params.get("rope_scaling")
            if self.rope_scaling["type"] == "longrope":
                scale = self.max_context_len / self.orig_context_len
                self.rope_scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.orig_context_len))

        # Vision params (Meta-specific)
        self.vision_chunk_size = -1
        self.vision_max_num_chunks = 4
        self.vision_num_cross_attention_layers = -1

        # Vision constants
        self.vision_dim = 1280
        self.vision_mlp_ratio = 4
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_act_layer = ttnn.UnaryOpType.GELU
        self.vision_dropout = 0.0
        self.vision_attn_n_heads = 16
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = 32
        self.vision_n_global_layers = 8
        self.vision_max_num_tiles = 4
        self.vision_patch_size = 14
        self.vision_in_channels = 3

    def __repr__(self):
        return f"""ModelArgs(
            dim={self.dim},
            n_layers={self.n_layers},
            n_heads={self.n_heads},
            n_kv_heads={self.n_kv_heads},
            vocab_size={self.vocab_size},
            multiple_of={self.multiple_of},
            norm_eps={self.norm_eps},
            rope_theta={self.rope_theta},
            rope_scaling_factor={self.rope_scaling_factor},
            rope_scaling={self.rope_scaling},
            max_batch_size={self.max_batch_size},
            max_seq_len={self.max_seq_len},
        )"""

    def reference_decoder(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        wrapper = HfDecoderWrapper(layer, self.head_dim)
        return wrapper


class HfDecoderWrapper(LightweightModule):
    def __init__(self, decoder, head_dim):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        result, self.past_key_values = self.decoder.forward(
            x,
            past_key_value=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        return result
