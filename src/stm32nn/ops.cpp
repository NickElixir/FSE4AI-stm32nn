#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>

#include "fp16.h"

at::Tensor round_fp16(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  const auto src = input.contiguous();
  auto output = at::empty(src.sizes(), src.options());
  stm32nn::round_fp16_values(src.data_ptr<float>(), output.data_ptr<float>(), src.numel());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("round_fp16", &round_fp16,
             "Round CPU float32 to IEEE binary16, ties to even, returned as float32",
             py::arg("input"));
}
