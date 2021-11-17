#include "ray_voxel_intersect.h"
#include <utility> 


void ray_voxel_intersect_wrapper(
  int device_id,
  int batch_size, int max_n,
  const float xyzmin, const float xyzmax, 
  const float voxel_num, const float voxel_size,
  const float *o, const float *v, 
  float *intersection, int *intersect_num, float *tns);

std::tuple<at::Tensor, at::Tensor, at::Tensor> ray_voxel_intersect(
    at::Tensor o, at::Tensor v, 
    const float xyzmin, const float xyzmax,
    const float voxel_num, const float voxel_size
){

  auto batch_size = o.size(0);
  int max_n = voxel_num * 3; // maximum possible intersections

  // buffer allocation
  at::Tensor intersection = 
      torch::zeros({batch_size, max_n, 3},
                    at::device(o.device()).dtype(at::ScalarType::Float));
  
  at::Tensor intersect_num = 
      torch::zeros({batch_size},
                    at::device(o.device()).dtype(at::ScalarType::Int));

  at::Tensor tns =
      torch::full({batch_size, max_n}, -1.0f, 
                   at::device(o.device()).dtype(at::ScalarType::Float));
  
  
  ray_voxel_intersect_wrapper(o.device().index(),
                              batch_size, max_n, 
                              xyzmin, xyzmax, voxel_num, voxel_size,
                              o.data_ptr <float>(), v.data_ptr <float>(), 
                              intersection.data_ptr <float>(), intersect_num.data_ptr <int>(),tns.data_ptr <float>());
  return std::make_tuple(intersection, intersect_num, tns);
}


void masked_intersect_wrapper(
  int device_id,
  int batch_size, int max_n, 
  const float xyzmin, const float xyzmax, 
  const float voxel_num, const float voxel_size, const float mask_scale,
  const float *o, const float *v, const bool *mask,
  float *intersection, int *intersect_num, float *tns);

std::tuple<at::Tensor, at::Tensor, at::Tensor> masked_intersect(
    at::Tensor o, at::Tensor v, at::Tensor mask,
    const float xyzmin, const float xyzmax,
    const float voxel_num, const float voxel_size, const float mask_scale
){

  auto batch_size = o.size(0);
  int max_n = voxel_num * 3; // maximum possible intersections
    
  // buffer allocation
  at::Tensor intersection = 
      torch::zeros({batch_size, max_n, 6},
                    at::device(o.device()).dtype(at::ScalarType::Float));
  
  at::Tensor intersect_num = 
      torch::zeros({batch_size},
                    at::device(o.device()).dtype(at::ScalarType::Int));

  at::Tensor tns =
      torch::full({batch_size, max_n}, -1.0f, 
                   at::device(o.device()).dtype(at::ScalarType::Float));
  
  
  masked_intersect_wrapper(o.device().index(),
                              batch_size, max_n, 
                              xyzmin, xyzmax, voxel_num, voxel_size, mask_scale,
                              o.data_ptr <float>(), v.data_ptr <float>(), mask.data_ptr <bool>(),
                              intersection.data_ptr <float>(), intersect_num.data_ptr <int>(),tns.data_ptr <float>());
  return std::make_tuple(intersection, intersect_num, tns);
}

