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

// Two independent arguments, each with the same correction as the helper.
#define TPC_EXP_F32_SAFE_PAIR(x0, x1, out0, out1) do { \
  float64 safe_pair_x0=(x0), safe_pair_x1=(x1); \
  float64 safe_pair_a0=v_exp_f32(safe_pair_x0), safe_pair_a1=v_exp_f32(safe_pair_x1); \
  float64 safe_pair_b0=v_exp_f32(safe_pair_x0 * 0.5f), safe_pair_b1=v_exp_f32(safe_pair_x1 * 0.5f); \
  safe_pair_b0 *= safe_pair_b0; safe_pair_b1 *= safe_pair_b1; \
  (out0)=v_f32_sel_grt_f32_b(safe_pair_a0*0.5f, safe_pair_b0*0.75f, safe_pair_b0, safe_pair_a0); \
  (out1)=v_f32_sel_grt_f32_b(safe_pair_a1*0.5f, safe_pair_b1*0.75f, safe_pair_b1, safe_pair_a1); \
} while (0)

#endif
