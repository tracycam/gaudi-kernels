// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Statistics are 64-row vectors; accumulators keep 3D [d, rows, heads] or
// 4D [d, rows, group, kv_heads] MME layout without reshape nodes.
// When prev_unscaled == 1 (first_chunk tile 1), old_output holds direct MME
// tile-0 P_0 @ V_0 and l0 is folded into wa = a * l0 in registers, eliminating
// a dedicated tile-0 TPC initialization kernel.
void main(tensor old_max, tensor old_sum, tensor old_output,
          tensor tile_max, tensor tile_sum, tensor tile_output,
          tensor new_max, tensor new_sum, tensor new_output,
          int prev_unscaled) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(old_output, 0), rows = get_dim_size(old_output, 1);
  const float ni = -1.0f / 0.0f;
  for (int kv_h = begin[3]; kv_h < end[3]; ++kv_h) {
    for (int head = begin[0]; head < end[0]; ++head) {
      for (int block = begin[1]; block < end[1]; ++block) {
        const int first = block * 64;
        int5 stat = {first, head, kv_h, 0, 0};
        float64 m0 = v_f32_ld_tnsr_b(stat, old_max), l0 = v_f32_ld_tnsr_b(stat, old_sum);
        float64 m1 = v_f32_ld_tnsr_b(stat, tile_max), l1 = v_f32_ld_tnsr_b(stat, tile_sum);
        float64 valid0 = v_f32_sel_grt_f32_b(l0, 0.f, m0, (float64)ni);
        float64 valid1 = v_f32_sel_grt_f32_b(l1, 0.f, m1, (float64)ni);
        float64 m = v_f32_max_b(valid0, valid1);
        float64 origin = v_f32_sel_eq_f32_b(m, ni, 0.f, m);
        float64 arg0 = v_f32_sel_grt_f32_b(l0, 0.f, m0 - origin, 0.f);
        float64 arg1 = v_f32_sel_grt_f32_b(l1, 0.f, m1 - origin, 0.f);
        float64 exp0, exp1;
        TPC_EXP_NEG_F32_PAIR(arg0, arg1, exp0, exp1);
        float64 a = v_f32_sel_grt_f32_b(l0, 0.f, exp0, 0.f);
        float64 b = v_f32_sel_grt_f32_b(l1, 0.f, exp1, 0.f);
        float64 al0 = a * l0;
        float64 wb = b * l1;
        float64 wa = prev_unscaled ? al0 : a;
        v_f32_st_tnsr(stat, new_max, m);
        v_f32_st_tnsr(stat, new_sum, al0 + wb);
        const int last = s_i32_min(rows - first, 64);
        int item = 0;
        if (width == 128) {
          for (; item + 1 < last; item += 2) {
            float64 aa0 = v_broadcast_element_f32(wa, item);
            float64 bb0 = v_broadcast_element_f32(wb, item);
            float64 aa1 = v_broadcast_element_f32(wa, item + 1);
            float64 bb1 = v_broadcast_element_f32(wb, item + 1);
            int5 p0 = {0, first + item, head, kv_h, 0};
            int5 p1 = {0, first + item + 1, head, kv_h, 0};
            float64 x0_lo = v_f32_ld_tnsr_b(p0, old_output), y0_lo = v_f32_ld_tnsr_b(p0, tile_output);
            p0[0] = 64;
            float64 x0_hi = v_f32_ld_tnsr_b(p0, old_output), y0_hi = v_f32_ld_tnsr_b(p0, tile_output);
            float64 x1_lo = v_f32_ld_tnsr_b(p1, old_output), y1_lo = v_f32_ld_tnsr_b(p1, tile_output);
            p1[0] = 64;
            float64 x1_hi = v_f32_ld_tnsr_b(p1, old_output), y1_hi = v_f32_ld_tnsr_b(p1, tile_output);
            p0[0] = 0;  v_f32_st_tnsr(p0, new_output, aa0 * x0_lo + bb0 * y0_lo);
            p0[0] = 64; v_f32_st_tnsr(p0, new_output, aa0 * x0_hi + bb0 * y0_hi);
            p1[0] = 0;  v_f32_st_tnsr(p1, new_output, aa1 * x1_lo + bb1 * y1_lo);
            p1[0] = 64; v_f32_st_tnsr(p1, new_output, aa1 * x1_hi + bb1 * y1_hi);
          }
          for (; item < last; ++item) {
            float64 aa = v_broadcast_element_f32(wa, item), bb = v_broadcast_element_f32(wb, item);
            int5 p = {0, first + item, head, kv_h, 0};
            float64 x_lo = v_f32_ld_tnsr_b(p, old_output), y_lo = v_f32_ld_tnsr_b(p, tile_output);
            p[0] = 64;
            float64 x_hi = v_f32_ld_tnsr_b(p, old_output), y_hi = v_f32_ld_tnsr_b(p, tile_output);
            p[0] = 0;  v_f32_st_tnsr(p, new_output, aa * x_lo + bb * y_lo);
            p[0] = 64; v_f32_st_tnsr(p, new_output, aa * x_hi + bb * y_hi);
          }
        } else {
          for (; item < last; ++item) {
            float64 aa = v_broadcast_element_f32(wa, item), bb = v_broadcast_element_f32(wb, item);
            for (int col = 0; col < width; col += 64) {
              int5 p = {col, first + item, head, kv_h, 0};
              float64 x = v_f32_ld_tnsr_b(p, old_output), y = v_f32_ld_tnsr_b(p, tile_output);
              v_f32_st_tnsr(p, new_output, aa * x + bb * y);
            }
          }
        }
      }
    }
  }
}
