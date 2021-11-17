#pragma once
#include <torch/extension.h>
#include <utility>

std::tuple<at::Tensor, at::Tensor, at::Tensor> ray_voxel_intersect(
    at::Tensor o, at::Tensor v, 
    const float xyzmin, const float xyzmax,
    const float voxel_num, const float voxel_size
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> masked_intersect(
    at::Tensor o, at::Tensor v, at::Tensor mask,
    const float xyzmin, const float xyzmax,
    const float voxel_num, const float voxel_size, const float mask_scale
);