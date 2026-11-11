// SPDX-License-Identifier: Apache-2.0
#include "norm_math.h"

void main(tensor x, tensor gamma, tensor norm_out, float epsilon) {
  const int width = get_dim_size(x, 0);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  bfloat128 saved[64];
  if ((width & 127) == 0 && width <= 8192) {
    for (int row = begin[0]; row < end[0]; ++row) {
      float128 ss = {0, 0};
      #pragma clang loop unroll_count(2)
      for (int k = 0; k < width; k += 128) {
        int5 pos = {k, row, 0, 0, 0};
        bfloat128 value = v_bf16_ld_tnsr_b(pos, x);
        saved[k >> 7] = value;
        ss = v_bf16_mac_acc32_b(value, value, ss);
      }
      float64 sum = norm_sum_squares(ss);
      float64 inverse = norm_refined_rsqrt(sum * (1.0f / (float)width) + epsilon);
      #pragma clang loop unroll_count(2)
      for (int k = 0; k < width; k += 128) {
        int5 pos = {k, row, 0, 0, 0}, g = {k, 0, 0, 0, 0};
        float128 value = v_convert_bf16_to_f32_all_b(saved[k >> 7]);
        float128 weight = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma));
        float128 normalized = {value.v1 * inverse * weight.v1, value.v2 * inverse * weight.v2};
        v_bf16_st_tnsr(pos, norm_out, convert_float128_to_bfloat128(normalized, SW_RHNE));
      }
    }
    return;
  }
  for (int row = begin[0]; row < end[0]; ++row) {
    float128 ss = {0, 0};
    for (int k = 0; k < width; k += 128) {
      int5 pos = {k, row, 0, 0, 0};
      int count = s_i32_min(width - k, 128) - 1;
      bfloat128 value = v_bf16_ld_tnsr_partial_b(pos, x, (char)count, 0);
      saved[k / 128] = value;
      ss = v_bf16_mac_acc32_b(value, value, ss);
    }
    float64 sum = norm_sum_squares(ss);
    float64 inverse = norm_refined_rsqrt(sum * (1.0f / (float)width) + epsilon);
    for (int k = 0; k < width; k += 128) {
      int5 pos = {k, row, 0, 0, 0}, g = {k, 0, 0, 0, 0};
      int count = s_i32_min(width - k, 128) - 1;
      float128 value = v_convert_bf16_to_f32_all_b(saved[k / 128]);
      float128 weight = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(g, gamma, (char)count, 0));
      float128 normalized = {value.v1 * inverse * weight.v1, value.v2 * inverse * weight.v2};
      v_bf16_st_tnsr_partial(pos, norm_out, convert_float128_to_bfloat128(normalized, SW_RHNE), (char)count, 0);
    }
  }
}
