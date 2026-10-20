// CPU kernels for the host side of an STM32N657 network check.
// The graded op is round_fp16:
//   y = x.to(torch.float16).to(torch.float32)
// IEEE binary16, round ties to even, stored back in float32.
// The other kernels score and prepare the same network. They are not the
// specification the README names.

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

void check_cpu_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Binary16 from binary32. Subnormal codes are multiples of 2^-24.
uint16_t float_to_half_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exp = (bits >> 23) & 0xffu;
  const uint32_t mant = bits & 0x7fffffu;

  if (exp == 0xffu) {
    if (mant == 0u) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half_mant = mant >> 13;
    if (half_mant == 0u) {
      half_mant = 1u;
    }
    return static_cast<uint16_t>(sign | 0x7c00u | half_mant);
  }
  if (exp == 0u) {
    return static_cast<uint16_t>(sign);
  }

  const int32_t half_exp = static_cast<int32_t>(exp) - 112;
  if (half_exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (half_exp <= 0) {
    if (half_exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t full = mant | 0x800000u;
    const int shift = 14 - half_exp;
    uint32_t truncated = full >> shift;
    const uint32_t rem_mask = (1u << shift) - 1u;
    const uint32_t halfway = 1u << (shift - 1);
    const uint32_t rem = full & rem_mask;
    if (rem > halfway || (rem == halfway && (truncated & 1u) != 0u)) {
      truncated += 1u;
    }
    if (truncated >= 0x400u) {
      return static_cast<uint16_t>(sign | (1u << 10));
    }
    return static_cast<uint16_t>(sign | truncated);
  }

  uint32_t half_mant = mant >> 13;
  const uint32_t round_bit = (mant >> 12) & 1u;
  const uint32_t sticky = mant & 0xfffu;
  int32_t out_exp = half_exp;
  if (round_bit != 0u && (sticky != 0u || (half_mant & 1u) != 0u)) {
    half_mant += 1u;
    if (half_mant == 0x400u) {
      half_mant = 0u;
      out_exp += 1;
      if (out_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(out_exp) << 10) | half_mant);
}

float half_bits_to_float(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
  const uint32_t exp = (half >> 10) & 0x1fu;
  uint32_t mant = half & 0x3ffu;
  uint32_t bits = 0;
  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      unsigned shift = 0;
      while ((mant & 0x400u) == 0u) {
        mant <<= 1;
        shift += 1;
      }
      mant &= 0x3ffu;
      const uint32_t exp32 = 113u - shift;
      bits = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 31u) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float out = 0.f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

torch::Tensor round_fp16_kernel(const torch::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  const torch::Tensor src = input.contiguous();
  auto output = torch::empty(src.sizes(), src.options());
  const float* in = src.data_ptr<float>();
  float* out = output.data_ptr<float>();
  const int64_t n = src.numel();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = half_bits_to_float(float_to_half_bits(in[i]));
  }
  return output;
}

torch::Tensor linear_kernel(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    bool relu) {
  check_cpu_f32(input, "input");
  check_cpu_f32(weight, "weight");
  TORCH_CHECK(input.dim() == 2, "linear input must have shape (N, in_features)");
  TORCH_CHECK(weight.dim() == 2, "linear weight must have shape (out_features, in_features)");
  TORCH_CHECK(input.size(1) == weight.size(1), "linear in_features do not match");
  const bool has_bias = bias.has_value() && bias->defined();
  if (has_bias) {
    check_cpu_f32(*bias, "bias");
    TORCH_CHECK(bias->dim() == 1, "linear bias must be 1-D");
    TORCH_CHECK(bias->size(0) == weight.size(0), "linear bias length does not match out_features");
  }

  const int64_t rows = input.size(0);
  const int64_t in_f = input.size(1);
  const int64_t out_f = weight.size(0);
  auto output = torch::empty({rows, out_f}, input.options());
  const float* x = input.data_ptr<float>();
  const float* w = weight.data_ptr<float>();
  const float* b = has_bias ? bias->data_ptr<float>() : nullptr;
  float* y = output.data_ptr<float>();

  for (int64_t n = 0; n < rows; ++n) {
    for (int64_t m = 0; m < out_f; ++m) {
      float acc = b != nullptr ? b[m] : 0.f;
      const float* w_row = w + m * in_f;
      const float* x_row = x + n * in_f;
      for (int64_t k = 0; k < in_f; ++k) {
        acc += x_row[k] * w_row[k];
      }
      y[n * out_f + m] = (relu && acc < 0.f) ? 0.f : acc;
    }
  }
  return output;
}

torch::Tensor conv2d_kernel(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool relu) {
  check_cpu_f32(input, "input");
  check_cpu_f32(weight, "weight");
  TORCH_CHECK(input.dim() == 4, "conv2d input must have shape (N, C, H, W)");
  TORCH_CHECK(weight.dim() == 4, "conv2d weight must have shape (out_channels, in_channels, kH, kW)");
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "conv2d stride must be positive");
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "conv2d padding must be non-negative");
  TORCH_CHECK(input.size(1) == weight.size(1), "conv2d in_channels do not match");
  const bool has_bias = bias.has_value() && bias->defined();
  if (has_bias) {
    check_cpu_f32(*bias, "bias");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0),
                "conv2d bias must have one value per output channel");
  }

  const int64_t n_batch = input.size(0);
  const int64_t in_c = input.size(1);
  const int64_t in_h = input.size(2);
  const int64_t in_w = input.size(3);
  const int64_t out_c = weight.size(0);
  const int64_t k_h = weight.size(2);
  const int64_t k_w = weight.size(3);
  const int64_t out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
  const int64_t out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
  TORCH_CHECK(out_h > 0 && out_w > 0, "conv2d output size is not positive");

  auto output = torch::empty({n_batch, out_c, out_h, out_w}, input.options());
  const float* x = input.data_ptr<float>();
  const float* w = weight.data_ptr<float>();
  const float* b = has_bias ? bias->data_ptr<float>() : nullptr;
  float* y = output.data_ptr<float>();

  for (int64_t n = 0; n < n_batch; ++n) {
    for (int64_t oc = 0; oc < out_c; ++oc) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          float acc = b != nullptr ? b[oc] : 0.f;
          for (int64_t ic = 0; ic < in_c; ++ic) {
            for (int64_t kh = 0; kh < k_h; ++kh) {
              const int64_t ih = oh * stride_h - pad_h + kh;
              if (ih < 0 || ih >= in_h) {
                continue;
              }
              for (int64_t kw = 0; kw < k_w; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw;
                if (iw < 0 || iw >= in_w) {
                  continue;
                }
                const float xv = x[((n * in_c + ic) * in_h + ih) * in_w + iw];
                const float wv = w[(((oc * in_c + ic) * k_h + kh) * k_w) + kw];
                acc += xv * wv;
              }
            }
          }
          y[((n * out_c + oc) * out_h + oh) * out_w + ow] = (relu && acc < 0.f) ? 0.f : acc;
        }
      }
    }
  }
  return output;
}

std::vector<double> compare_kernel(const torch::Tensor& logits, const torch::Tensor& reference) {
  check_cpu_f32(logits, "logits");
  check_cpu_f32(reference, "reference");
  TORCH_CHECK(logits.sizes() == reference.sizes(), "compare tensors must have the same shape");
  TORCH_CHECK(logits.dim() >= 1 && logits.size(-1) > 0, "compare expects a non-empty last dimension");

  const int64_t width = logits.size(-1);
  const int64_t rows = logits.numel() / width;
  const float* a = logits.data_ptr<float>();
  const float* b = reference.data_ptr<float>();
  double gap_sum = 0.0;
  double cos_sum = 0.0;
  int64_t disagree = 0;
  for (int64_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    int64_t arg_a = 0;
    int64_t arg_b = 0;
    float max_a = a[r * width];
    float max_b = b[r * width];
    for (int64_t c = 0; c < width; ++c) {
      const float da = a[r * width + c];
      const float db = b[r * width + c];
      const double d = static_cast<double>(da) - static_cast<double>(db);
      acc += d * d;
      if (da > max_a) {
        max_a = da;
        arg_a = c;
      }
      if (db > max_b) {
        max_b = db;
        arg_b = c;
      }
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int64_t c = 0; c < width; ++c) {
      const double da = a[r * width + c];
      const double db = b[r * width + c];
      dot += da * db;
      na += da * da;
      nb += db * db;
    }
    if (na == 0.0 && nb == 0.0) {
      cos_sum += 1.0;
    } else if (na == 0.0 || nb == 0.0) {
      cos_sum += 0.0;
    } else {
      cos_sum += dot / std::sqrt(na * nb);
    }
    gap_sum += acc;
    disagree += arg_a != arg_b;
  }
  const double n = static_cast<double>(rows);
  return {gap_sum / n, static_cast<double>(disagree) / n, cos_sum / n};
}

torch::Tensor to_nhwc_kernel(const torch::Tensor& input) {
  check_cpu_f32(input, "input");
  TORCH_CHECK(input.dim() == 4, "to_nhwc expects NCHW with shape (N, C, H, W)");
  return input.permute({0, 2, 3, 1}).contiguous();
}

torch::Tensor to_nchw_kernel(const torch::Tensor& input) {
  check_cpu_f32(input, "input");
  TORCH_CHECK(input.dim() == 4, "to_nchw expects NHWC with shape (N, H, W, C)");
  return input.permute({0, 3, 1, 2}).contiguous();
}

torch::Tensor quantize_per_channel_kernel(const torch::Tensor& input, int64_t axis) {
  check_cpu_f32(input, "input");
  TORCH_CHECK(input.dim() >= 1, "quantize_per_channel expects at least 1 dimension");
  if (axis < 0) {
    axis += input.dim();
  }
  TORCH_CHECK(axis >= 0 && axis < input.dim(), "quantize_per_channel axis is out of range");

  auto output = torch::empty_like(input);
  const float* in = input.data_ptr<float>();
  float* out = output.data_ptr<float>();
  const int64_t n_ch = input.size(axis);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < input.dim(); ++d) {
    inner *= input.size(d);
  }
  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) {
    outer *= input.size(d);
  }
  const int64_t plane = n_ch * inner;

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t ch = 0; ch < n_ch; ++ch) {
      float amax = 0.f;
      for (int64_t i = 0; i < inner; ++i) {
        const float v = std::fabs(in[o * plane + ch * inner + i]);
        if (v > amax) {
          amax = v;
        }
      }
      const float scale = std::max(amax, 1e-12f) / 127.f;
      for (int64_t i = 0; i < inner; ++i) {
        const int64_t idx = o * plane + ch * inner + i;
        float q = std::nearbyint(in[idx] / scale);
        if (q > 127.f) {
          q = 127.f;
        } else if (q < -127.f) {
          q = -127.f;
        }
        out[idx] = q * scale;
      }
    }
  }
  return output;
}

torch::Tensor max_pool2d_kernel(
    const torch::Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w) {
  check_cpu_f32(input, "input");
  TORCH_CHECK(input.dim() == 4, "max_pool2d expects NCHW");
  TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "max_pool2d kernel must be positive");
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "max_pool2d stride must be positive");
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "max_pool2d padding must be non-negative");
  const int64_t n_batch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t in_h = input.size(2);
  const int64_t in_w = input.size(3);
  const int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  const int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
  TORCH_CHECK(out_h > 0 && out_w > 0, "max_pool2d output size is not positive");
  auto output = torch::empty({n_batch, channels, out_h, out_w}, input.options());
  const float* x = input.data_ptr<float>();
  float* y = output.data_ptr<float>();
  for (int64_t n = 0; n < n_batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          float best = -std::numeric_limits<float>::infinity();
          bool any = false;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= in_h) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - pad_w + kw;
              if (iw < 0 || iw >= in_w) {
                continue;
              }
              const float v = x[((n * channels + c) * in_h + ih) * in_w + iw];
              if (!any || v > best) {
                best = v;
                any = true;
              }
            }
          }
          TORCH_CHECK(any, "max_pool2d window is empty");
          y[((n * channels + c) * out_h + oh) * out_w + ow] = best;
        }
      }
    }
  }
  return output;
}

bool is_large_prime(int64_t n) {
  if (n <= 64) {
    return false;
  }
  if (n % 2 == 0) {
    return false;
  }
  for (int64_t d = 3; d * d <= n; d += 2) {
    if (n % d == 0) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> export_issues_kernel(
    const std::vector<torch::Tensor>& weights,
    const torch::Tensor& sample) {
  std::vector<std::string> issues;
  for (size_t i = 0; i < weights.size(); ++i) {
    const torch::Tensor& w = weights[i];
    TORCH_CHECK(w.device().is_cpu(), "weight ", i, " must be a CPU tensor");
    if (w.scalar_type() == torch::kFloat16) {
      issues.push_back("weight " + std::to_string(i) + " is float16; Cube.AI rejects an FP16 graph");
    }
    if (w.dim() >= 1 && is_large_prime(w.size(0))) {
      issues.push_back(
          "weight " + std::to_string(i) + " has " + std::to_string(w.size(0)) +
          " channels on axis 0, a prime above 64; Neural-ART has failed to map some prime channel counts");
    }
  }
  TORCH_CHECK(sample.device().is_cpu(), "sample must be a CPU tensor");
  if (sample.scalar_type() != torch::kFloat32) {
    issues.push_back("sample is not float32; this project divides a uint8 image by 255");
  } else if (sample.numel() > 0) {
    const float peak = sample.max().item<float>();
    if (peak > 1.5f) {
      issues.push_back("sample values exceed 1; expected pixels divided by 255");
    }
  }
  return issues;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "round_fp16",
      &round_fp16_kernel,
      "Round each finite float32 element to IEEE binary16, ties to even, and store it as float32",
      py::arg("input"));
  module.def(
      "linear",
      &linear_kernel,
      "FP32 linear layer, input-feature accumulation, optional fused ReLU",
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias") = c10::nullopt,
      py::arg("relu") = false);
  module.def(
      "conv2d",
      &conv2d_kernel,
      "FP32 NCHW conv2d, channel-then-kernel accumulation, optional fused ReLU",
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias") = c10::nullopt,
      py::arg("stride_h") = 1,
      py::arg("stride_w") = 1,
      py::arg("pad_h") = 0,
      py::arg("pad_w") = 0,
      py::arg("relu") = false);
  module.def(
      "compare",
      &compare_kernel,
      "Mean squared Euclidean logit gap, argmax disagreement, and mean cosine",
      py::arg("logits"),
      py::arg("reference"));
  module.def("to_nhwc", &to_nhwc_kernel, "NCHW to contiguous NHWC", py::arg("input"));
  module.def("to_nchw", &to_nchw_kernel, "NHWC to contiguous NCHW", py::arg("input"));
  module.def(
      "quantize_per_channel",
      &quantize_per_channel_kernel,
      "Symmetric int8 per channel, returned as dequantized FP32",
      py::arg("input"),
      py::arg("axis") = 0);
  module.def(
      "max_pool2d",
      &max_pool2d_kernel,
      "FP32 NCHW max pool",
      py::arg("input"),
      py::arg("kernel_h"),
      py::arg("kernel_w"),
      py::arg("stride_h") = 1,
      py::arg("stride_w") = 1,
      py::arg("pad_h") = 0,
      py::arg("pad_w") = 0);
  module.def(
      "export_issues",
      &export_issues_kernel,
      "Problems that would make this graph a bad STM32N657 export",
      py::arg("weights"),
      py::arg("sample"));
}
