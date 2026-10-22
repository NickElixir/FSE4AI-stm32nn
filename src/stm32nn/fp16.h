// Portable IEEE binary16 conversion shared by the host and Cortex-M55 build.
#pragma once

#include <stdint.h>
#include <string.h>

namespace stm32nn {

// Binary16 from binary32. Subnormal codes are multiples of 2^-24.
inline uint16_t float_to_half_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
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

inline float half_bits_to_float(uint16_t half) {
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
  memcpy(&out, &bits, sizeof(out));
  return out;
}


inline void round_fp16_values(const float* input, float* output, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    output[i] = half_bits_to_float(float_to_half_bits(input[i]));
  }
}

}  // namespace stm32nn
