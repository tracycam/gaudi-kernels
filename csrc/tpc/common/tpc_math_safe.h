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

// Pure-VPU sigmoid(x) (0 LUTs, handles +/-Inf without overflow/cancellation,
// propagates NaN).
static inline float64 tpc_sigmoid_f32_safe(float64 x) {
  uint64 xabs = (uint64)x & 0x7fffffffu;
  float64 ax = (float64)xabs;
  float64 e = tpc_exp_f32_safe(-ax);
  float64 inv = tpc_rcp_nonneg_f32(1.0f + e);
  float64 res = v_f32_sel_geq_f32_b(x, 0.0f, inv, e * inv);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, res);
}

// Pure-VPU tanh(x) (0 LUTs): for |x| < 0.625f uses odd Remez minimax
// x + x * x^2 * P(x^2) (< 1 ULP, exact 0 -> 0, exact odd symmetry); for
// |x| >= 0.625f uses 1 - 2 * exp(-2|x|) / (1 + exp(-2|x|)), restoring sign
// and propagating NaN.
static inline float64 tpc_tanh_f32_safe(float64 x) {
  uint64 xbits = (uint64)x;
  uint64 sign = xbits & 0x80000000u;
  uint64 xabs = xbits & 0x7fffffffu;
  float64 ax = (float64)xabs;

  float64 u = ax * ax;
  float64 p = 2.298947191e-03f;
  p = v_f32_mac_b(p, u, -8.350013755e-03f, 0);
  p = v_f32_mac_b(p, u, 2.177099697e-02f, 0);
  p = v_f32_mac_b(p, u, -5.395955220e-02f, 0);
  p = v_f32_mac_b(p, u, 1.333330572e-01f, 0);
  p = v_f32_mac_b(p, u, -3.333333433e-01f, 0);
  float64 t_small = v_f32_mac_b(ax * u, p, ax, 0);

  float64 e = tpc_exp_f32_safe(-2.0f * ax);
  float64 t_large = 1.0f - 2.0f * e * tpc_rcp_nonneg_f32(1.0f + e);

  float64 mag = v_f32_sel_less_f32_b(ax, 0.625f, t_small, t_large);
  float64 res = (float64)(sign | (uint64)mag);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, res);
}

// Pure-VPU tanh-approximate GELU (0 LUTs): computes x * sigmoid(2 * inner)
// to avoid 1 + tanh(-|u|) catastrophic cancellation on negative inputs.
static inline float64 tpc_gelu_tanh_f32_safe(float64 x) {
  uint64 xabs = (uint64)x & 0x7fffffffu;
  float64 inner2 = v_f32_mac_b(0.044715f * x, x * x, x, 0) * 1.595769167f;
  float64 g = x * tpc_sigmoid_f32_safe(inner2);
  g = v_f32_sel_leq_f32_b(x, -15.0f, 0.0f, g);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, g);
}

// Pure-VPU degree-7 Cody-Waite exp(hi + lo) for hi + lo <= 0 (used by gelu_erf
// tail so Veltkamp-split -x^2/2 avoids mantissa cancellation).
static inline float64 tpc_exp7_neg_hi_lo_f32(float64 hi, float64 lo) {
  float64 x = hi + lo;
  float64 xc = v_f32_max_b(x, -87.33654475f);
  int64 ni = v_convert_f32_to_i32_b(xc * TPC_LOG2E_F32, SW_RHNE);
  float64 nf = v_convert_i32_to_f32_b(ni, SW_RHNE);
  float64 r = v_f32_mac_b(nf, -TPC_LN2_HI_F32, hi, 0) + lo;
  r = v_f32_mac_b(nf, -TPC_LN2_LO_F32, r, 0);
  float64 m = 1.992423640e-04f;
  m = v_f32_mac_b(m, r, 1.394858002e-03f, 0);
  m = v_f32_mac_b(m, r, 8.333248086e-03f, 0);
  m = v_f32_mac_b(m, r, 4.166621715e-02f, 0);
  m = v_f32_mac_b(m, r, 1.666666716e-01f, 0);
  m = v_f32_mac_b(m, r, 5.000000000e-01f, 0);
  m = v_f32_mac_b(m, r, 1.000000000e+00f, 0);
  m = v_f32_mac_b(m, r, 1.000000000e+00f, 0);
  float64 res = (float64)v_i32_add_b((int64)m, v_i32_shl_b(ni, 23));
  return v_f32_sel_less_f32_b(x, -87.33654475f, 0.0f, res);
}

// Pure-VPU exact GELU (0.5 * x * (1 + erf(x / sqrt(2))), 0 LUTs, <= ~2 ULP).
// Separates [0, 1.25), (-1.25, 0), and Veltkamp-split erfcx tail |x| >= 1.25
// to avoid catastrophic cancellation near -1.0 and preserve +/-Inf and NaN.
static inline float64 tpc_gelu_erf_f32_safe(float64 x) {
  uint64 xabs = (uint64)x & 0x7fffffffu;
  float64 ax = (float64)xabs;

  // [0, 1.25): x * (0.5 + 0.5 * z * P_erf(z^2)), z = x / sqrt(2)
  float64 z = x * 0.7071067811865475f;
  float64 z2 = z * z;
  float64 pe = 8.642171451e-05f;
  pe = v_f32_mac_b(pe, z2, -8.204489131e-04f, 0);
  pe = v_f32_mac_b(pe, z2, 5.206267349e-03f, 0);
  pe = v_f32_mac_b(pe, z2, -2.686143108e-02f, 0);
  pe = v_f32_mac_b(pe, z2, 1.128373072e-01f, 0);
  pe = v_f32_mac_b(pe, z2, -3.761263490e-01f, 0);
  pe = v_f32_mac_b(pe, z2, 1.128379226e+00f, 0);
  float64 g_pos_small = x * v_f32_mac_b(0.5f * z, pe, 0.5f, 0);

  // (-1.25, 0): x * 0.5 * erfc(|x| / sqrt(2)) evaluated directly around w = |x| - 0.625
  float64 w = ax - 0.625f;
  float64 pn = 2.182397475e-05f;
  pn = v_f32_mac_b(pn, w, -3.146455565e-04f, 0);
  pn = v_f32_mac_b(pn, w, -2.099265112e-05f, 0);
  pn = v_f32_mac_b(pn, w, 3.194629448e-03f, 0);
  pn = v_f32_mac_b(pn, w, -2.212235006e-03f, 0);
  pn = v_f32_mac_b(pn, w, -2.229810134e-02f, 0);
  pn = v_f32_mac_b(pn, w, 3.332887590e-02f, 0);
  pn = v_f32_mac_b(pn, w, 1.025502458e-01f, 0);
  pn = v_f32_mac_b(pn, w, -3.281609714e-01f, 0);
  pn = v_f32_mac_b(pn, w, 2.659855187e-01f, 0);
  float64 g_neg_small = x * pn;

  float64 g_small = v_f32_sel_geq_f32_b(x, 0.0f, g_pos_small, g_neg_small);

  // |x| >= 1.25: exp(-|x|^2/2) * (0.5 * |x| * erfcx(|x|/sqrt(2)))
  float64 ax_c = v_f32_min_b(ax, 13.5f);
  float64 t = tpc_rcp_nonneg_f32(ax_c + 2.0f);
  float64 q = 2.211819916e+02f;
  q = v_f32_mac_b(q, t, -2.786744385e+02f, 0);
  q = v_f32_mac_b(q, t, 9.837424469e+01f, 0);
  q = v_f32_mac_b(q, t, 9.145334244e+00f, 0);
  q = v_f32_mac_b(q, t, -4.267615795e+00f, 0);
  q = v_f32_mac_b(q, t, -3.426405668e+00f, 0);
  q = v_f32_mac_b(q, t, -1.606377959e+00f, 0);
  q = v_f32_mac_b(q, t, -3.986293077e-01f, 0);
  q = v_f32_mac_b(q, t, -3.422186410e-06f, 0);
  q = v_f32_mac_b(q, t, 3.989422917e-01f, 0);
  float64 xh = (float64)((uint64)ax_c & 0xfffff000u);
  float64 xl = ax_c - xh;
  float64 hi = -0.5f * (xh * xh);
  float64 lo = -0.5f * (xl * (ax_c + xh));
  float64 e = tpc_exp7_neg_hi_lo_f32(hi, lo);
  float64 corr = v_f32_sel_geq_f32_b(ax, 13.25f, 0.0f, e * q);
  float64 g_tail = v_f32_sel_geq_f32_b(x, 0.0f, x - corr, -corr);

  float64 res = v_f32_sel_less_f32_b(ax, 1.25f, g_small, g_tail);
  res = v_f32_sel_leq_f32_b(x, -13.25f, 0.0f, res);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, res);
}

// Pure-VPU rsqrt(x) (0 LUTs, avoids Gaudi2 SFU v_rsqrt_f32 16 KiB VLM cap).
static inline float64 tpc_rsqrt_f32_safe(float64 x) {
  uint64 xbits = (uint64)x;
  uint64 xabs = xbits & 0x7fffffffu;
  float64 safe = v_f32_sel_grt_f32_b(x, 0.0f, v_f32_sel_eq_u32_b(xbits, 0x7f800000u, 1.0f, x), 1.0f);
  float64 y = (float64)v_i32_sub_b((int64)0x5f3759df, v_i32_shr_b((int64)safe, 1));
  float64 hx = 0.5f * safe;
  y = y * v_f32_mac_b(-hx * y, y, 1.5f, 0);
  y = y * v_f32_mac_b(-hx * y, y, 1.5f, 0);
  y = y * v_f32_mac_b(-hx * y, y, 1.5f, 0);
  y = y * v_f32_mac_b(-hx * y, y, 1.5f, 0);
  y = v_f32_sel_eq_f32_b(x, 0.0f, (float64)(uint64)0x7f800000u, y);
  y = v_f32_sel_eq_u32_b(xbits, 0x7f800000u, 0.0f, y);
  y = v_f32_sel_less_f32_b(x, 0.0f, (float64)(uint64)0x7fc00000u, y);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, y);
}

// Pure-VPU sqrt(x) (0 LUTs, avoids Gaudi2 SFU v_sqrt_f32 16 KiB VLM cap).
static inline float64 tpc_sqrt_f32_safe(float64 x) {
  uint64 xbits = (uint64)x;
  uint64 xabs = xbits & 0x7fffffffu;
  float64 s = x * tpc_rsqrt_f32_safe(x);
  s = v_f32_sel_eq_f32_b(x, 0.0f, x, s);
  s = v_f32_sel_eq_u32_b(xbits, 0x7f800000u, x, s);
  s = v_f32_sel_less_f32_b(x, 0.0f, (float64)(uint64)0x7fc00000u, s);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, s);
}

#endif
