// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Merge two FP32 (max, sum, unnormalised output) triples. Empty tiles have a
// zero sum and contribute nothing, including when both inputs are empty.
void main(tensor old_max, tensor old_sum, tensor old_output,
          tensor tile_max, tensor tile_sum, tensor tile_output,
          tensor new_max, tensor new_sum, tensor new_output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(old_output, 0);
  const float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int row = begin[1]; row < end[1]; ++row) {
      int5 stat = {row, head, 0, 0, 0};
      float m0 = s_f32_ld_g(gen_addr(stat, old_max)), l0 = s_f32_ld_g(gen_addr(stat, old_sum));
      float m1 = s_f32_ld_g(gen_addr(stat, tile_max)), l1 = s_f32_ld_g(gen_addr(stat, tile_sum));
      float m = l0 > 0.f ? (l1 > 0.f ? (m0 > m1 ? m0 : m1) : m0) : (l1 > 0.f ? m1 : ni);
      float64 a = 0.f, b = 0.f;
      if (l0 > 0.f) a = tpc_exp_f32_safe((float64)(m0 - m));
      if (l1 > 0.f) b = tpc_exp_f32_safe((float64)(m1 - m));
      float64 sum = a * l0 + b * l1;
      v_f32_st_tnsr_partial(stat, new_max, (float64)m, 0, 0);
      v_f32_st_tnsr_partial(stat, new_sum, sum, 0, 0);
      for (int col = 0; col < width; col += 64) {
        int5 p = {col, row, head, 0, 0};
        int count = s_i32_min(width - col, 64) - 1;
        float64 o0 = v_f32_ld_tnsr_partial_b(p, old_output, (char)count, 0);
        float64 o1 = v_f32_ld_tnsr_partial_b(p, tile_output, (char)count, 0);
        v_f32_st_tnsr_partial(p, new_output, a * o0 + b * o1, (char)count, 0);
      }
    }
  }
}
