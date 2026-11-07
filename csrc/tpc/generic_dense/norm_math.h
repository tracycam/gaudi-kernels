// SPDX-License-Identifier: Apache-2.0
#ifndef GAUDIRT_GENERIC_DENSE_NORM_MATH_H
#define GAUDIRT_GENERIC_DENSE_NORM_MATH_H

static inline float64 norm_sum_squares(float128 sums) {
  // Preserve low bits lost when joining the two accumulators and reducing lanes.
  float64 lanes = sums.v1 + sums.v2;
  float64 first_error = (sums.v1 - lanes) + sums.v2;
  float64 second_error = (sums.v2 - lanes) + sums.v1;
  float64 join_error = v_f32_sel_geq_f32_b(sums.v1, sums.v2, first_error, second_error);
  float64 total = v_f32_reduce_add(lanes);
  float64 residual = v_f32_reduce_add((lanes - total * (1.0f / 64.0f)) + join_error);
  residual = v_f32_sel_eq_f32_b(total, (float64)((uint64)0x7f800000), 0.f, residual);
  return total + residual;
}

static inline float64 norm_refined_rsqrt(float64 value) {
  float64 estimate = v_rsqrt_f32(value);
  float64 square = estimate * estimate;
  float64 square_error = v_f32_mac_b(estimate, estimate, -square);
  float64 error = v_f32_mac_b(-value, square, 1.f) - value * square_error;
  float64 refined = v_f32_mac_b(estimate * 0.5f, error, estimate);
  return v_f32_sel_eq_f32_b(value, (float64)((uint64)0x7f800000), 0.f, refined);
}

#endif
