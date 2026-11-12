// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

void main(tensor exponent, tensor sums, tensor result) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int keys = get_dim_size(exponent, 0);
  for (int head = begin[0]; head < end[0]; ++head) {
    int5 stat = {0, head, 0, 0, 0};
    float64 sum = v_f32_ld_tnsr_b(stat, sums);
    float64 inv = tpc_rcp_nonneg_f32(sum);
    int i = 0;
    for (; i + 255 < keys; i += 256) {
      int5 p = {i, 0, head, 0, 0};
      float64 a0 = v_f32_ld_tnsr_b(p, exponent) * inv; p[0] = i + 64;
      float64 b0 = v_f32_ld_tnsr_b(p, exponent) * inv; p[0] = i + 128;
      float64 a1 = v_f32_ld_tnsr_b(p, exponent) * inv; p[0] = i + 192;
      float64 b1 = v_f32_ld_tnsr_b(p, exponent) * inv;
      float128 pair0 = {a0, b0}, pair1 = {a1, b1};
      p[0] = i;       v_bf16_st_tnsr(p, result, convert_float128_to_bfloat128(pair0, SW_LINEAR | SW_RHNE));
      p[0] = i + 128; v_bf16_st_tnsr(p, result, convert_float128_to_bfloat128(pair1, SW_LINEAR | SW_RHNE));
    }
    for (; i < keys; i += 128) {
      int5 p = {i, 0, head, 0, 0};
      float64 a = v_f32_ld_tnsr_b(p, exponent) * inv; p[0] = i + 64;
      float64 b = v_f32_ld_tnsr_b(p, exponent) * inv; p[0] = i;
      float128 pair = {a, b};
      v_bf16_st_tnsr(p, result, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
    }
  }
}
