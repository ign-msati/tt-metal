// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_op.hpp"

#include "ttnn/run_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "untilize_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks) {
    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    if (nblocks <= ncores) {
        ncores = nblocks;
    } else {
        uint32_t nblocks_per_core = std::ceil((float)nblocks / ncores);
        ncores = std::ceil((float)nblocks / nblocks_per_core);
    }
    return ncores;
}
}  // namespace untilize_helpers

void Untilize::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.padded_shape()[-1] % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(
        (input_tensor_a.physical_volume() / input_tensor_a.padded_shape()[-1]) % TILE_HEIGHT == 0,
        "Height must be evenly divisible into tiles");

    if (this->sub_core_grids.has_value()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            this->use_multicore == true,
            "sub_core_grid implementation only supported when use_multicore flag argument is set to true");
    }

    if (!this->use_multicore) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::SINGLE_BANK,
            "Input memory layout must be interleaved or sharded");
        TT_FATAL(
            this->output_mem_config.memory_layout() != TensorMemoryLayout::SINGLE_BANK,
            "Output memory layout must be interleaved or sharded");

        if (input_tensor_a.is_sharded()) {
            std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
            TT_FATAL(
                input_tensor_a.padded_shape()[-1] % input_shard_shape[1] == 0,
                "Uneven input shard shape not supported");
            TT_FATAL(
                (input_tensor_a.physical_volume() / input_tensor_a.padded_shape()[-1]) % input_shard_shape[0] == 0,
                "Uneven input shard shape not supported");
        }
        if (this->output_mem_config.is_sharded()) {
            std::array<uint32_t, 2> output_shard_shape = this->output_mem_config.shard_spec().value().shape;
            TT_FATAL(
                input_tensor_a.padded_shape()[-1] % output_shard_shape[1] == 0,
                "Uneven output shard shape not supported");
            TT_FATAL(
                (input_tensor_a.physical_volume() / input_tensor_a.padded_shape()[-1]) % output_shard_shape[0] == 0,
                "Uneven output shard shape not supported");
        }
    } else if (input_tensor_a.memory_config().is_sharded()) {
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(
                this->output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(), "Error");
        }
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1, "Error");
        }
        TT_FATAL(this->use_multicore == true, "Error");
    } else if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->use_multicore == true, "Error");
        TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        uint32_t ntiles = input_tensor_a.physical_volume() / TILE_HW;
        uint32_t ntiles_per_block = input_tensor_a.padded_shape()[-1] / TILE_WIDTH;
        uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block);
        auto num_cores =
            untilize_helpers::get_num_cores(input_tensor_a.device()->compute_with_storage_grid_size(), nblocks);
        uint32_t fused_height = input_tensor_a.physical_volume() / input_tensor_a.padded_shape()[-1];
        TT_FATAL(fused_height % num_cores == 0, "Error");
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    }
}

std::vector<ttnn::TensorSpec> Untilize::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    if (!this->use_multicore) {
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                this->output_mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()))};
    }

    if (this->output_mem_config.is_sharded()) {
        if (input_tensor.memory_config().is_sharded()) {
            auto mem_config = this->output_mem_config.with_shard_spec(input_tensor.memory_config().shard_spec());
            return {TensorSpec(
                input_tensor.logical_shape(),
                TensorLayout::fromPaddedShape(
                    output_dtype,
                    PageConfig(Layout::ROW_MAJOR),
                    mem_config,
                    input_tensor.logical_shape(),
                    input_tensor.padded_shape()))};
        }

        uint32_t ntiles = input_tensor.physical_volume() / TILE_HW;
        uint32_t ntiles_per_block = input_tensor.padded_shape()[-1] / TILE_WIDTH;
        uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block);
        auto num_cores =
            untilize_helpers::get_num_cores(input_tensor.device()->compute_with_storage_grid_size(), nblocks);
        auto shard_grid = tt::tt_metal::num_cores_to_corerangeset(
            num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
        uint32_t fused_height = input_tensor.physical_volume() / input_tensor.padded_shape()[-1];
        std::array<uint32_t, 2> shard_shape = {fused_height / num_cores, input_tensor.padded_shape()[-1]};
        ShardSpec shard_spec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR};
        auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()))};
    }

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            this->output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks Untilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (!this->use_multicore) {
        return detail::untilize_single_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (this->sub_core_grids.has_value()) {
        // If sub_core_grids parameter is provided, use custom sub_core_grid implementation instead
        // of the standard multicore implementation or the block multicore implementation
        return detail::untilize_multi_core_sub_core_grids(
            input_tensor_a,
            output_tensor,
            this->use_pack_untilize,
            this->fp32_dest_acc_en,
            this->sub_core_grids.value());
    }
    if (!this->enough_space_height && !input_tensor_a.is_sharded() && !output_tensor.is_sharded()) {
        return detail::untilize_multi_core_block(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }

    return detail::untilize_multi_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
}

}  // namespace ttnn::operations::data_movement
