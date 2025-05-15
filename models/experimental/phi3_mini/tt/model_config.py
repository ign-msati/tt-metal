import ttnn


class TtPhi3MiniKernelConfigs:
    # Keys to be used by the different modules of Grok
    OP_KEYS = (
        # Attention
        "ATTN_WEIGHTS",
        "ATTN_CACHE_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "LM_HEAD_OUTPUT",
        "ATTN_W_LAYOUT",
    )

    def __init__(self, device=None):
        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

        self.model_mem_configs = {}
        # Update memory configs (By default weights->DRAM, activations->L1)
        self.model_mem_configs.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_mem_configs.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        # Set configurations for sharded type
        self.model_mem_configs["WIDTH_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
        )
        self.model_mem_configs["HEIGHT_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1
        )
        self.model_mem_configs["BLOCK_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1
        )

        # Create sharded memory configs for different ops
        self.model_mem_configs["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_mem_configs["Q_TRANSPOSE_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_mem_configs["ATTN_BATCHED_MM_OUTPUT_MEMCFG"] = cached_lambda(
            lambda padded_layer_past_len: ttnn.create_sharded_memory_config(
                shape=(32, padded_layer_past_len),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        )

        self.model_mem_configs["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        shard_height = 32
        shard_width_hidden_dim_across_32_cores = 3072 // 32  # hidden_size = 6144
        self.model_mem_configs["SHARDED_NORM_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width_hidden_dim_across_32_cores),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_mem_configs["SHARDED_NORM_OUTPUT_MEMCFG"] = self.model_mem_configs["SHARDED_NORM_INPUT_MEMCFG"]

        # Create program configs for the different ttnn matmul ops
        # TODO: update for 6144 not 4096?
        self.model_mem_configs["ROT_MAT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        self.model_mem_configs["ATTN_BATCHED_SOFTMAX_PROGCFG"] = cached_lambda(
            lambda padded_layer_past_len: ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
                subblock_w=1,
                block_h=1,  # Shard_height // 32,
                block_w=padded_layer_past_len // 32,  # Dynamic
            )
        )

        self.model_mem_configs["GATE_MM_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=24,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        self.model_mem_configs["QKV_MM_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=6,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_mem_configs["SCORES_BATCHED_MM_PROGCFG"] = cached_lambda(
            lambda p: ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=p,
            )
        )

        self.model_mem_configs["VALUES_BATCHED_MM_PROGCFG"] = cached_lambda(
            lambda p: ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=p,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
            )
        )

        self.model_mem_configs["LM_HEAD_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=1,
            per_core_N=3,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_mem_configs["FF1_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 6144 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 32768
            fuse_batch=True,
            fused_activation=(ttnn.UnaryOpType.GELU, True),  # FIXME: GET THIS DOCUMENTED
            mcast_in0=True,
        )

        self.model_mem_configs["FF3_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,  # K = 6144 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 32768
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_mem_configs["FF2_OUTPUT_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            # Issue #8959: Increasing subblock to 2 results in hangs -> Potentially related to di/dt hangs.
            out_subblock_w=3,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=3,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 6144
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_mem_configs["OUTPUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(7, 6),  # TODO Hanging with full coreGrid (8,8)
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.model_mem_configs["SHARDED_NORM_PRGM_CFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=3,
            block_h=shard_height // 32,
            block_w=shard_width_hidden_dim_across_32_cores // 32,
            inplace=False,
        )

        if device is not None:
            grid_size = device.compute_with_storage_grid_size()
            # TODO Lower max grid size (used by MLP) to avoid hangs
            self.max_grid_size = ttnn.CoreGrid(y=7, x=6)  # (y,x)  (y=7, x=8)
            # self.max_grid_size = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)  # (y,x)  (y=7, x=8)
            self.core_grid_attention = (
                ttnn.CoreGrid(y=4, x=8) if (4 <= grid_size.y and 8 <= grid_size.x) else self.max_grid_size
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # LoFi
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_attn_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # HiFi2
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_output_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # HiFi2?
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Create Compute kernel configs
        self.model_mem_configs["ROT_MAT_COMPUTE_KERNEL_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def get_model_mem_configs(self):
        return self.model_mem_configs

    def get_compute_kernel_config(self):
        return self.compute_kernel_config

    def get_compute_kernel_attn_config(self):
        return self.compute_kernel_attn_config

    def get_compute_kernel_output_config(self):
        return self.compute_kernel_output_config


def cached_lambda(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper
