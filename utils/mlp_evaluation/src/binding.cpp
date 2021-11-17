#include "mlp_eval.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mlp_eval", &mlp_eval);
  m.def("upload_weight", &upload_weight);
}