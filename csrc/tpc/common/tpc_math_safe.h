// SPDX-License-Identifier: Apache-2.0
#ifndef GAUDIRT_TPC_MATH_SAFE_H
#define GAUDIRT_TPC_MATH_SAFE_H

#define TPC_LOG2E_F32  1.4426950408889634f
#define TPC_LN2_HI_F32 6.9314575195e-01f
#define TPC_LN2_LO_F32 1.4286067653e-06f

// Pure-VPU 2-FMA Cody-Waite + Estrin degree-6 minimax expf (<= 2 ULP across
// [-87, 88.3], exact P(0)=1.0f): avoids Gaudi2 SFU/LUT -k*ln(2) discontinuity
// (P-TPC-01), 16 KiB VLM cap (P-TPC-03), and single-constant x*log2(e) range
// reduction drift (P-NUM-43) in 14 VPU ops with low VRF pressure.
static inline float64 tpc_exp_f32_safe(float64 x) {
  float64 xc = v_f32_min_b(v_f32_max_b(x, -87.33654475f), 88.37626266f);
  int64 ni = v_convert_f32_to_i32_b(xc * TPC_LOG2E_F32, SW_RHNE);
  float64 nf = v_convert_i32_to_f32_b(ni, SW_RHNE);
  float64 r = v_f32_mac_b(nf, -TPC_LN2_HI_F32, xc, 0);
  r = v_f32_mac_b(nf, -TPC_LN2_LO_F32, r, 0);
  float64 r2 = r * r, r4 = r2 * r2;
  float64 q0 = v_f32_mac_b(r2, 1.3875052100e-03f, r * 8.3728292957e-03f + 4.1667692363e-02f, 0);
  float64 q1 = v_f32_mac_b(r2, r * 1.6666479409e-01f + 4.9999994040e-01f, r + 1.0f, 0);
  float64 mant = v_f32_mac_b(r4, q0, q1, 0);
  float64 res = (float64)v_i32_add_b((int64)mant, v_i32_shl_b(ni, 23));
  return v_f32_sel_less_f32_b(x, -87.33654475f, 0.0f, res);
}

// Non-positive argument fast path (x <= 0, e.g. shifted softmax scores):
// skips upper clamp and interleaves 2 or 4 vectors to hide VPU FMA latency.
#define TPC_EXP_NEG_F32_PAIR(x0, x1, out0, out1) do { \
  float64 _x0 = v_f32_max_b((x0), -87.33654475f), _x1 = v_f32_max_b((x1), -87.33654475f); \
  int64 _n0 = v_convert_f32_to_i32_b(_x0 * TPC_LOG2E_F32, SW_RHNE); \
  int64 _n1 = v_convert_f32_to_i32_b(_x1 * TPC_LOG2E_F32, SW_RHNE); \
  float64 _nf0 = v_convert_i32_to_f32_b(_n0, SW_RHNE), _nf1 = v_convert_i32_to_f32_b(_n1, SW_RHNE); \
  float64 _r0 = v_f32_mac_b(_nf0, -TPC_LN2_HI_F32, _x0, 0); \
  float64 _r1 = v_f32_mac_b(_nf1, -TPC_LN2_HI_F32, _x1, 0); \
  _r0 = v_f32_mac_b(_nf0, -TPC_LN2_LO_F32, _r0, 0); \
  _r1 = v_f32_mac_b(_nf1, -TPC_LN2_LO_F32, _r1, 0); \
  _n0 = v_i32_shl_b(_n0, 23); _n1 = v_i32_shl_b(_n1, 23); \
  float64 _s0 = _r0 * _r0, _s1 = _r1 * _r1; \
  float64 _f0 = _s0 * _s0, _f1 = _s1 * _s1; \
  float64 _a0 = v_f32_mac_b(_s0, 1.3875052100e-03f, _r0 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _a1 = v_f32_mac_b(_s1, 1.3875052100e-03f, _r1 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _b0 = v_f32_mac_b(_s0, _r0 * 1.6666479409e-01f + 4.9999994040e-01f, _r0 + 1.0f, 0); \
  float64 _b1 = v_f32_mac_b(_s1, _r1 * 1.6666479409e-01f + 4.9999994040e-01f, _r1 + 1.0f, 0); \
  _b0 = v_f32_mac_b(_f0, _a0, _b0, 0); \
  _b1 = v_f32_mac_b(_f1, _a1, _b1, 0); \
  (out0) = (float64)v_i32_add_b((int64)_b0, _n0); \
  (out1) = (float64)v_i32_add_b((int64)_b1, _n1); \
} while (0)

#define TPC_EXP_NEG_F32_QUAD(x0, x1, x2, x3, out0, out1, out2, out3) do { \
  float64 _x0 = v_f32_max_b((x0), -87.33654475f), _x1 = v_f32_max_b((x1), -87.33654475f); \
  float64 _x2 = v_f32_max_b((x2), -87.33654475f), _x3 = v_f32_max_b((x3), -87.33654475f); \
  int64 _n0 = v_convert_f32_to_i32_b(_x0 * TPC_LOG2E_F32, SW_RHNE); \
  int64 _n1 = v_convert_f32_to_i32_b(_x1 * TPC_LOG2E_F32, SW_RHNE); \
  int64 _n2 = v_convert_f32_to_i32_b(_x2 * TPC_LOG2E_F32, SW_RHNE); \
  int64 _n3 = v_convert_f32_to_i32_b(_x3 * TPC_LOG2E_F32, SW_RHNE); \
  float64 _nf0 = v_convert_i32_to_f32_b(_n0, SW_RHNE), _nf1 = v_convert_i32_to_f32_b(_n1, SW_RHNE); \
  float64 _nf2 = v_convert_i32_to_f32_b(_n2, SW_RHNE), _nf3 = v_convert_i32_to_f32_b(_n3, SW_RHNE); \
  float64 _r0 = v_f32_mac_b(_nf0, -TPC_LN2_HI_F32, _x0, 0); \
  float64 _r1 = v_f32_mac_b(_nf1, -TPC_LN2_HI_F32, _x1, 0); \
  float64 _r2 = v_f32_mac_b(_nf2, -TPC_LN2_HI_F32, _x2, 0); \
  float64 _r3 = v_f32_mac_b(_nf3, -TPC_LN2_HI_F32, _x3, 0); \
  _r0 = v_f32_mac_b(_nf0, -TPC_LN2_LO_F32, _r0, 0); \
  _r1 = v_f32_mac_b(_nf1, -TPC_LN2_LO_F32, _r1, 0); \
  _r2 = v_f32_mac_b(_nf2, -TPC_LN2_LO_F32, _r2, 0); \
  _r3 = v_f32_mac_b(_nf3, -TPC_LN2_LO_F32, _r3, 0); \
  _n0 = v_i32_shl_b(_n0, 23); _n1 = v_i32_shl_b(_n1, 23); \
  _n2 = v_i32_shl_b(_n2, 23); _n3 = v_i32_shl_b(_n3, 23); \
  float64 _s0 = _r0 * _r0, _s1 = _r1 * _r1, _s2 = _r2 * _r2, _s3 = _r3 * _r3; \
  float64 _f0 = _s0 * _s0, _f1 = _s1 * _s1, _f2 = _s2 * _s2, _f3 = _s3 * _s3; \
  float64 _a0 = v_f32_mac_b(_s0, 1.3875052100e-03f, _r0 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _a1 = v_f32_mac_b(_s1, 1.3875052100e-03f, _r1 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _a2 = v_f32_mac_b(_s2, 1.3875052100e-03f, _r2 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _a3 = v_f32_mac_b(_s3, 1.3875052100e-03f, _r3 * 8.3728292957e-03f + 4.1667692363e-02f, 0); \
  float64 _b0 = v_f32_mac_b(_s0, _r0 * 1.6666479409e-01f + 4.9999994040e-01f, _r0 + 1.0f, 0); \
  float64 _b1 = v_f32_mac_b(_s1, _r1 * 1.6666479409e-01f + 4.9999994040e-01f, _r1 + 1.0f, 0); \
  float64 _b2 = v_f32_mac_b(_s2, _r2 * 1.6666479409e-01f + 4.9999994040e-01f, _r2 + 1.0f, 0); \
  float64 _b3 = v_f32_mac_b(_s3, _r3 * 1.6666479409e-01f + 4.9999994040e-01f, _r3 + 1.0f, 0); \
  _b0 = v_f32_mac_b(_f0, _a0, _b0, 0); _b1 = v_f32_mac_b(_f1, _a1, _b1, 0); \
  _b2 = v_f32_mac_b(_f2, _a2, _b2, 0); _b3 = v_f32_mac_b(_f3, _a3, _b3, 0); \
  (out0) = (float64)v_i32_add_b((int64)_b0, _n0); \
  (out1) = (float64)v_i32_add_b((int64)_b1, _n1); \
  (out2) = (float64)v_i32_add_b((int64)_b2, _n2); \
  (out3) = (float64)v_i32_add_b((int64)_b3, _n3); \
} while (0)

#define TPC_EXP_F32_SAFE_PAIR(x0, x1, out0, out1) do { \
  (out0) = tpc_exp_f32_safe(x0); \
  (out1) = tpc_exp_f32_safe(x1); \
} while (0)

// Pure-VPU IEEE-754 magic initial guess + 3-step Newton-Raphson reciprocal
// (<= 1 ULP across normal positive FP32, exact 1.0f at x=1.0f, 0 SFU LUTs so
// full 64 KiB VLM stays unlocked per P-TPC-03). Returns 0.0f when x <= 0.0f.
static inline float64 tpc_rcp_nonneg_f32(float64 x) {
  float64 safe = v_f32_sel_grt_f32_b(x, 0.0f, x, 1.0f);
  float64 y = (float64)v_i32_sub_b((int64)0x7EF127EA, (int64)safe);
  y = v_f32_mac_b(y, 1.0f - safe * y, y, 0);
  y = v_f32_mac_b(y, 1.0f - safe * y, y, 0);
  y = v_f32_mac_b(y, 1.0f - safe * y, y, 0);
  return v_f32_sel_grt_f32_b(x, 0.0f, y, 0.0f);
}

#endif
