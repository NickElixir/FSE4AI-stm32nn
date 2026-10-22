// Compile the same conversion loop for the deployment CPU, without PyTorch.
#include "fp16.h"

static_assert(sizeof(float) == 4, "IEEE binary32 requires a 32-bit float");

extern "C" void stm32nn_round_fp16(const float* input, float* output, int64_t count) {
  stm32nn::round_fp16_values(input, output, count);
}
