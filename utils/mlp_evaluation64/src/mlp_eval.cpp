#include "mlp_eval.h"
#include <utility> 


void mlp_eval_wrapper(
  int batch_size,
  const float* coord,
  const float* voxels,
  const float* v,
  float* rgba,
  bool* mask
  );

void upload_weight_wrapper(
  int device_id,
  int map_size,
  const float* params,
  const int* voxel_map
);

void mlp_eval(
    at::Tensor rgba,
    at::Tensor coord, at::Tensor voxels,
    at::Tensor v, at::Tensor mask
){

  auto batch_size = v.size(0)*v.size(1);
  
  mlp_eval_wrapper(batch_size,
                    coord.data_ptr <float>(), voxels.data_ptr <float>(),
                    v.data_ptr <float>(),
                    rgba.data_ptr <float>(),mask.data_ptr<bool>());
}

void upload_weight(
    int device_id,
    at::Tensor params,
    at::Tensor voxel_map) {
    
    int map_size = voxel_map.size(0);
    
    upload_weight_wrapper(
        device_id,
        map_size,
        params.data_ptr<float>(),
        voxel_map.data_ptr<int>()
    );
}
