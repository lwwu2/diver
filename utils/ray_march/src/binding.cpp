#include "ray_march.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aabb_intersect", &aabb_intersect);
    m.def("ray_march", &ray_march);
}