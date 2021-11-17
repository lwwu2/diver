#include "ray_voxel_intersect.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_voxel_intersect", &ray_voxel_intersect);
    m.def("masked_intersect", &masked_intersect);
}