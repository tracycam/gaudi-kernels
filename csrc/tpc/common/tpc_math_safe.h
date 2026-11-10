// SPDX-License-Identifier: Apache-2.0
#ifndef GAUDIRT_TPC_MATH_SAFE_H
#define GAUDIRT_TPC_MATH_SAFE_H

// Pure-VPU Cody-Waite degree-6 polynomial exp (v_exp_cephes_f32): avoids
// Gaudi2 SFU/LUT -k*ln(2) discontinuity (P-TPC-01) and 16 KiB VLM cap (P-TPC-03)
// in 18 VPU ops instead of 62.
static inline float64 tpc_exp_f32_safe(float64 x) {
  return v_exp_cephes_f32(x);
}

#define TPC_EXP_F32_SAFE_PAIR(x0, x1, out0, out1) do { \
  (out0) = v_exp_cephes_f32(x0); \
  (out1) = v_exp_cephes_f32(x1); \
} while (0)

#endif
