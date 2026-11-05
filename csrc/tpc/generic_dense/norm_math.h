// SPDX-License-Identifier: Apache-2.0
#ifndef GAUDIRT_GENERIC_DENSE_NORM_MATH_H
#define GAUDIRT_GENERIC_DENSE_NORM_MATH_H

static inline float64 norm_refined_rsqrt(float64 value) {
  float64 estimate = v_rsqrt_f32(value);
  float64 square = estimate * estimate;
  float64 square_error = v_f32_mac_b(estimate, estimate, -square);
  float64 error = v_f32_mac_b(-value, square, 1.f) - value * square_error;
  float64 refined = v_f32_mac_b(estimate * 0.5f, error, estimate);
  return v_f32_sel_eq_f32_b(value, (float64)((uint64)0x7f800000), 0.f, refined);
}

#endif
