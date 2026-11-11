// SPDX-License-Identifier: Apache-2.0
// Dedicated 1-output group-128 dynamic OCP E4M3FN activation quant-dequant producer
// for SRAM-bundled BF16 GEMM (x0 when KARATSUBA_32 is unset, x0 + 32 * (x_q - x0) when set).
#include "activation_quant_common.h"

static inline void emit_act_vec(tensor output, float128 x, float64 scale, float64 rcp, int col, int row) {
  float128 w = {fp8_ocp_dequant_vec(x.v1, scale, rcp),
                fp8_ocp_dequant_vec(x.v2, scale, rcp)};
  bfloat128 h = convert_float128_to_bfloat128(w, SW_RHNE);
#ifdef KARATSUBA_32
  float128 hw = v_convert_bf16_to_f32_all_b(h);
  float128 wp = {v_f32_mac_b(w.v1 - hw.v1, 32.0f, hw.v1),
                 v_f32_mac_b(w.v2 - hw.v2, 32.0f, hw.v2)};
  h = convert_float128_to_bfloat128(wp, SW_RHNE);
#endif
  int5 dst = {col, row, 0, 0, 0};
  v_bf16_st_tnsr(dst, output, h);
}

void main(tensor input, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(input, 0) / 128;
  int rows = get_dim_size(input, 1);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int row = r0; row < r_end; ++row) {
      int g0 = begin[0] * 4;
      int g_end = s_i32_min(end[0] * 4, groups);
      int g = g0;
      for (; g + 1 < g_end; g += 2) {
        int c0 = g * 128, c1 = c0 + 128;
        int5 p0 = {c0, row, 0, 0, 0}, p1 = {c1, row, 0, 0, 0};
        float128 x0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0, input));
        float128 x1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1, input));
        float64 m0 = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x0.v1 & 0x7fffffffu),
                                                  (float64)((uint64)x0.v2 & 0x7fffffffu)));
        float64 m1 = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x1.v1 & 0x7fffffffu),
                                                  (float64)((uint64)x1.v2 & 0x7fffffffu)));
        float64 mp = v_f32_mov_dual_group_all_b(m1, 0xffffffff, 1, 1, 1, 1, MkWrA(0, 3, 0, 0), m0);
        // Substitute 1.0f denominator ONLY when a 128-element group is identically zero.
        mp = v_f32_sel_eq_f32_b(mp, 0.0f, 1.0f, mp);
        float64 sp = fp8_activation_scale(mp);
        float64 rcpp = fp8_exact_positive_rcp(sp);

        float64 s0 = v_f32_mov_dual_group_all_b(sp, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), sp);
        float64 r0v = v_f32_mov_dual_group_all_b(rcpp, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), rcpp);
        emit_act_vec(output, x0, s0, r0v, c0, row);

        float64 s1 = v_f32_mov_dual_group_all_b(sp, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), sp);
        float64 r1v = v_f32_mov_dual_group_all_b(rcpp, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), rcpp);
        emit_act_vec(output, x1, s1, r1v, c1, row);
      }
      if (g < g_end) {
        int col = g * 128;
        int5 pos = {col, row, 0, 0, 0};
        float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pos, input));
        float64 maximum = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x.v1 & 0x7fffffffu),
                                                       (float64)((uint64)x.v2 & 0x7fffffffu)));
        maximum = v_f32_sel_eq_f32_b(maximum, 0.0f, 1.0f, maximum);
        float64 scale = fp8_activation_scale(maximum);
        float64 rcp = fp8_exact_positive_rcp(scale);
        emit_act_vec(output, x, scale, rcp, col, row);
      }
    }
  }
}
