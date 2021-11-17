#pragma once
#include <torch/extension.h>
#include <utility>

void mlp_eval(
    at::Tensor rgba,
    at::Tensor coord,
    at::Tensor voxels,
    at::Tensor voxel_map,
    at::Tensor v,
    at::Tensor mask
);

void mlp_eval64(
    at::Tensor rgba,
    at::Tensor coord,
    at::Tensor voxels,
    at::Tensor voxel_map,
    at::Tensor v,
    at::Tensor mask
);


void upload_weight(
    int device_id,
    at::Tensor params,
    at::Tensor voxel_chunk,
    at::Tensor chunk_map
);