#pragma once
#include <torch/extension.h>
#include <utility>

void mlp_eval(
    at::Tensor rgba,
    at::Tensor coord,
    at::Tensor voxels,
    at::Tensor v,
    at::Tensor mask
);

void upload_weight(
    int device_id,
    at::Tensor params,
    at::Tensor voxel_map
);