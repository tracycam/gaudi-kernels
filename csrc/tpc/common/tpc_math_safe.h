// SPDX-License-Identifier: Apache-2.0
#ifndef GAUDIRT_TPC_MATH_SAFE_H
#define GAUDIRT_TPC_MATH_SAFE_H

// Gaudi2's exp may double at a range boundary. Compare with the squared
// half-argument result, which avoids the same discontinuity.
static inline float64 tpc_exp_f32_safe(float64 x) {
  float64 value = v_exp_f32(x);
  float64 half_exp = v_exp_f32(x * 0.5f);
  float64 squared = half_exp * half_exp;
  return v_f32_sel_grt_f32_b(value * 0.5f, squared * 0.75f, squared, value);
}

#endif
