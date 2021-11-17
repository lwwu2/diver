#include "mlp_eval.h"
#include <utility> 


void mlp_eval_wrapper(
  int batch_size,
  const float* coord,
  const float* voxels,
  const int* voxel_map,
  const float* v,
  float* rgba,
  bool* mask
  );

void mlp_eval64_wrapper(
  int batch_size,
  const float* coord,
  const float* voxels,
  const int* voxel_map,
  const float* v,
  float* rgba,
  bool* mask
  );

void upload_weight_wrapper(
  int device_id,
  int chunk_num, int chunk_scale, int chunk_size,
  const float* params,
  const float* voxel_chunk,
  const int* chunk_map
);

void mlp_eval(
    at::Tensor rgba,
    at::Tensor coord, at::Tensor voxels, at::Tensor voxel_map,
    at::Tensor v, at::Tensor mask
){

  auto batch_size = v.size(0)*v.size(1);
  
  mlp_eval_wrapper(batch_size,
                    coord.data_ptr <float>(), voxels.data_ptr <float>(), voxel_map.data_ptr<int>(),
                    v.data_ptr <float>(),
                    rgba.data_ptr <float>(),mask.data_ptr<bool>());
}

void mlp_eval64(
    at::Tensor rgba,
    at::Tensor coord, at::Tensor voxels, at::Tensor voxel_map,
    at::Tensor v, at::Tensor mask
){

  auto batch_size = v.size(0)*v.size(1);
  
  mlp_eval_wrapper(batch_size,
                    coord.data_ptr <float>(), voxels.data_ptr <float>(), voxel_map.data_ptr<int>(),
                    v.data_ptr <float>(),
                    rgba.data_ptr <float>(),mask.data_ptr<bool>());
}

void upload_weight(
    int device_id,
    at::Tensor params,
    at::Tensor voxel_chunk,
    at::Tensor chunk_map) {
    
    int chunk_scale = chunk_map.size(0);
    int chunk_num = voxel_chunk.size(0);
    int chunk_size = voxel_chunk.size(1);
    
    upload_weight_wrapper(
        device_id,
        chunk_num,chunk_scale,chunk_size,
        params.data_ptr<float>(),
        voxel_chunk.data_ptr<float>(),
        chunk_map.data_ptr<int>()
    );
}
