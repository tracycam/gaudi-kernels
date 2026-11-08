// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Statistics are 64-row vectors; accumulators keep the MME [d, rows, heads] layout.
void main(tensor old_max, tensor old_sum, tensor old_output,
          tensor tile_max, tensor tile_sum, tensor tile_output,
          tensor new_max, tensor new_sum, tensor new_output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(old_output, 0), rows = get_dim_size(old_output, 1);
  const float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int block = begin[1]; block < end[1]; ++block) {
      const int first = block * 64;
      int5 stat = {first,head,0,0,0};
      float64 m0 = v_f32_ld_tnsr_b(stat, old_max), l0 = v_f32_ld_tnsr_b(stat, old_sum);
      float64 m1 = v_f32_ld_tnsr_b(stat, tile_max), l1 = v_f32_ld_tnsr_b(stat, tile_sum);
      float64 valid0 = v_f32_sel_grt_f32_b(l0, 0.f, m0, (float64)ni);
      float64 valid1 = v_f32_sel_grt_f32_b(l1, 0.f, m1, (float64)ni);
      float64 m = v_f32_max_b(valid0, valid1);
      float64 origin = v_f32_sel_eq_f32_b(m, ni, 0.f, m);
      float64 arg0 = v_f32_sel_grt_f32_b(l0, 0.f, m0 - origin, 0.f);
      float64 arg1 = v_f32_sel_grt_f32_b(l1, 0.f, m1 - origin, 0.f);
      float64 a = v_f32_sel_grt_f32_b(l0, 0.f, tpc_exp_f32_safe(arg0), 0.f);
      float64 b = v_f32_sel_grt_f32_b(l1, 0.f, tpc_exp_f32_safe(arg1), 0.f);
      v_f32_st_tnsr(stat, new_max, m);
      v_f32_st_tnsr(stat, new_sum, a * l0 + b * l1);
      const int last = s_i32_min(rows - first, 64);
      for (int item = 0; item < last; ++item) {
        float64 aa = v_broadcast_element_f32(a, item), bb = v_broadcast_element_f32(b * l1, item);
        int5 p = {0,first + item,head,0,0};
        float64 x = v_f32_ld_tnsr_b(p, old_output), y = v_f32_ld_tnsr_b(p, tile_output);
        v_f32_st_tnsr(p, new_output, aa * x + bb * y);
        if (width == 128) {
          p[0] = 64;
          x = v_f32_ld_tnsr_b(p, old_output); y = v_f32_ld_tnsr_b(p, tile_output);
          v_f32_st_tnsr(p, new_output, aa * x + bb * y);
        }
      }
    }
  }
}
