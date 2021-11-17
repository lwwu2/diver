#pragma once
#include <torch/extension.h>
#include <utility>

void aabb_intersect(
    at::Tensor o, at::Tensor v, at::Tensor center,  
    at::Tensor finish, at::Tensor rgba, 
    at::Tensor octrees,
    int voxel_num
);

void ray_march(
    at::Tensor o, at::Tensor v, 
    at::Tensor mask, at::Tensor finish,
    int voxel_num
);

