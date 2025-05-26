# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ShardTensor2dMesh
from models.common.lightweightmodule import LightweightModule
from models.utility_functions import nearest_32
from models.experimental.phi3_mini_may_ver_5.tt.phi3_mini_common import precompute_freqs
from models.tt_transformers.tt.common import (
    get_rot_transformation_mat,
    gather_cos_sin,
)


def compute_gather_cos_sin(dhead, end, theta, rope_scaling_factor, position_ids):
    rope_scaling_long_factor = torch.tensor(rope_scaling_factor["long_factor"], dtype=torch.float32)
    rope_scaling_short_factor = torch.tensor(rope_scaling_factor["short_factor"], dtype=torch.float32)
    cos_long, sin_long = precompute_freqs(dhead, end, rope_scaling_long_factor, theta=theta)
    cos_short, sin_short = precompute_freqs(dhead, end, rope_scaling_short_factor, theta=theta)
    cos_long, sin_long = gather_cos_sin(position_ids, cos_long, sin_long)
    cos_short, sin_short = gather_cos_sin(position_ids, cos_short, sin_short)
    return cos_long, sin_long, cos_short, sin_short


class RotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling_factor: dict,  # use None to disable rope scaling
        orig_context_len: int,  # only used if scaling enabled
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(self.batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()
        self.orig_context_len = orig_context_len

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_long_matrix, sin_long_matrix, cos_short_matrix, sin_short_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len,
            theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            position_ids=torch.arange(max_seq_len),
        )

        self.cos_long_matrix = ttnn.from_torch(
            cos_long_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_long_matrix = ttnn.from_torch(
            sin_long_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.cos_short_matrix = ttnn.from_torch(
            cos_short_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_short_matrix = ttnn.from_torch(
            sin_short_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        self.batch_grid = (
            ttnn.CoreGrid(y=4, x=8)
            if ttnn.get_arch_name() == "blackhole"
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )
        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            batch_size,
            1,
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=(
                ShardTensor2dMesh(
                    device,
                    dims=(None, 2) if (self.num_devices == 32 and batch_size > 1) else (None, None),
                    mesh_shape=list(device.shape),
                )
                if self.is_mesh_device
                else None
            ),
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )

        return rot_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        ################################################33
        # float_rot_idxs = ttnn.to_layout(rot_idxs, ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        # is_larger = ((ttnn.max(float_rot_idxs)) > (self.orig_context_len - 1))
        # is_smaller = ((ttnn.max(float_rot_idxs)) < (self.orig_context_len - 1))
        # cos = self.cos_long_matrix * is_larger + self.cos_short_matrix * is_smaller
        # sin = self.sin_long_matrix * is_larger + self.sin_short_matrix * is_smaller
        # ttnn.deallocate(float_rot_idxs)
        # ttnn.deallocate(is_larger)
        # ttnn.deallocate(is_smaller)
        # cos = ttnn.embedding(rot_idxs, cos, layout=embedding_layout)  # [1, batch, head_dim]
        # sin = ttnn.embedding(rot_idxs, sin, layout=embedding_layout)  # [1, batch, head_dim]
        ###################################################
        # # self.cos_short_matrix =[seq_len,96]

        float_rot_idxs = ttnn.to_layout(rot_idxs, ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        max_rot_id = ttnn.max(float_rot_idxs)
        ttnn.deallocate(float_rot_idxs)

        is_larger = ttnn.gt(max_rot_id, (self.orig_context_len - 1))
        is_smaller = ttnn.lt(max_rot_id, (self.orig_context_len - 1))
        ttnn.deallocate(max_rot_id)

        cos = self.cos_long_matrix * is_larger + self.cos_short_matrix * is_smaller
        sin = self.sin_long_matrix * is_larger + self.sin_short_matrix * is_smaller
        ttnn.deallocate(is_larger)
        ttnn.deallocate(is_smaller)

        cos = ttnn.embedding(rot_idxs, cos, layout=embedding_layout)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, sin, layout=embedding_layout)  # [1, batch, head_dim]
        # ####################################################
        # # if isinstance(position_idxs, ttnn.Tensor):
        # ############33
        # # if isinstance(position_idxs, ttnn.Tensor):
        # #     torch_position_idxs = ttnn.to_torch(position_idxs)
        # # else:
        # #     torch_position_idxs = position_idxs
        # ##################3
        # # FIXME: Might not be the best for cases where there is large difference between max and min position ids in the batches
        # if (torch.max(torch_position_idxs) + 1) > self.orig_context_len:
        #     cos = ttnn.embedding(rot_idxs, self.cos_long_matrix, layout=embedding_layout)  # [1, batch, head_dim]
        #     sin = ttnn.embedding(rot_idxs, self.sin_long_matrix, layout=embedding_layout)  # [1, batch, head_dim]
        # else:
        # cos = ttnn.embedding(rot_idxs, self.cos_short_matrix, layout=embedding_layout)  # [1, batch, head_dim]
        # sin = ttnn.embedding(rot_idxs, self.sin_short_matrix, layout=embedding_layout)  # [1, batch, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], head_dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
