// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

static inline float64 ewise_exp_f32(float64 x) {
  float64 e = tpc_exp_f32_safe(x);
  uint64 xabs = (uint64)x & 0x7fffffffu;
  e = v_f32_sel_grt_f32_b(x, 88.722839f, (float64)(uint64)0x7f800000u, e);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, e);
}

static inline float64 ewise_silu_f32(float64 x) {
  uint64 xabs = (uint64)x & 0x7fffffffu;
  float64 s = x * tpc_sigmoid_f32_safe(x);
  s = v_f32_sel_leq_f32_b(x, -88.0f, 0.0f, s);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, s);
}

static inline float64 ewise_clamp_f32(float64 x, float alpha, float beta) {
  uint64 xabs = (uint64)x & 0x7fffffffu;
  float64 c = v_f32_min_b(v_f32_max_b(x, alpha), beta);
  return v_f32_sel_grt_u32_b(xabs, 0x7f800000u, x, c);
}

#ifdef EWISE_BF16
#define EWISE_LOOP(EXPR) \
  for (int _idx = start; _idx < end; ++_idx) { \
    const int _row = _idx / chunks; \
    const int _col = (_idx - _row * chunks) * 128; \
    const char _cnt = (char)(s_i32_min(128, width - _col) - 1); \
    int5 _p = {_col, _row, 0, 0, 0}; \
    float128 _v = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_p, input, _cnt, 0), SW_LINEAR); \
    float64 x = _v.v1; \
    float64 _o0 = (EXPR); \
    x = _v.v2; \
    float64 _o1 = (EXPR); \
    float128 _out = {_o0, _o1}; \
    v_bf16_st_tnsr_partial(_p, output, convert_float128_to_bfloat128(_out, SW_LINEAR | SW_RHNE), _cnt, 0); \
  }
#else
#define EWISE_LOOP(EXPR) \
  for (int _idx = start; _idx < end; ++_idx) { \
    const int _row = _idx / chunks; \
    const int _col = (_idx - _row * chunks) * 64; \
    const char _cnt = (char)(s_i32_min(64, width - _col) - 1); \
    int5 _p = {_col, _row, 0, 0, 0}; \
    float64 x = v_f32_ld_tnsr_partial_b(_p, input, _cnt, 0); \
    float64 _out = (EXPR); \
    v_f32_st_tnsr_partial(_p, output, _out, _cnt, 0); \
  }
#endif

void main(tensor input, tensor output, int op_code, float alpha, float beta) {
  const int width = get_dim_size(input, 0);
  const int rows = get_dim_size(input, 1);
#ifdef EWISE_BF16
  const int step = 128;
#else
  const int step = 64;
#endif
  const int chunks = (width + step - 1) / step;
  const int total = rows * chunks;

  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const int per = total / ntid;
  const int rem = total % ntid;
  const int start = tid * per + s_i32_min(tid, rem);
  const int end = start + per + (tid < rem ? 1 : 0);

  const float inv_alpha = alpha != 0.0f ? (1.0f / alpha) : 0.0f;

  switch (op_code) {
    case 0: EWISE_LOOP(ewise_exp_f32(x)); break;
    case 1: EWISE_LOOP(tpc_tanh_f32_safe(x)); break;
    case 2: EWISE_LOOP(tpc_sigmoid_f32_safe(x)); break;
    case 3: EWISE_LOOP(ewise_silu_f32(x)); break;
    case 4: EWISE_LOOP(tpc_gelu_erf_f32_safe(x)); break;
    case 5: EWISE_LOOP(tpc_gelu_tanh_f32_safe(x)); break;
    case 6: EWISE_LOOP(alpha * tpc_tanh_f32_safe(x * inv_alpha)); break;
    case 7: EWISE_LOOP(ewise_clamp_f32(x, alpha, beta)); break;
    case 8: EWISE_LOOP(tpc_rsqrt_f32_safe(x)); break;
    case 9: EWISE_LOOP(tpc_sqrt_f32_safe(x)); break;
    case 10: EWISE_LOOP(-x); break;
    case 11: EWISE_LOOP((float64)((uint64)x & 0x7fffffffu)); break;
    default: break;
  }
}
