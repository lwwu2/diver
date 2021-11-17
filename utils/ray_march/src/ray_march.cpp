#include "ray_march.h"
#include <utility> 


void aabb_intersect_wrapper(
    const int batch_size,
    float* o, const float* v, const float* center,
    bool* finish, float* rgba, 
    const bool* octrees,
    int voxel_num
);

void ray_march_wrapper(
    const int batch_size,
    float* o, const float* v,
    const bool* mask, bool* finish, 
    int voxel_num
);

void aabb_intersect(
    at::Tensor o, at::Tensor v, at::Tensor center,
    at::Tensor finish, 
    at::Tensor rgba,
    at::Tensor octrees,
    int voxel_num
){

  auto batch_size = v.size(0)*v.size(1);  
  aabb_intersect_wrapper(
                         batch_size,
                         o.data_ptr<float>(), v.data_ptr<float>(), 
                         center.data_ptr<float>(),
                         finish.data_ptr<bool>(), 
                         rgba.data_ptr<float>(),
                         octrees.data_ptr<bool>(),
                         voxel_num);
}

void ray_march(
    at::Tensor o, at::Tensor v, at::Tensor mask, at::Tensor finish,
    int voxel_num
){

  // HitxCxHxW
  auto batch_size = v.size(0)*v.size(1);
  ray_march_wrapper(batch_size,  
                    o.data_ptr<float>(), v.data_ptr<float>(), 
                    mask.data_ptr<bool>(), finish.data_ptr<bool>(),
                    voxel_num);

}

