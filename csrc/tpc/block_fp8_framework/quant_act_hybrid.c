// SPDX-License-Identifier: Apache-2.0
// Fused group-128 dynamic FP8 E4M3FN activation quantization producing both
// 1K BF16 main component (out_1k) and 2K FP8_143 residual ([x_q / s_row, 256 * dx / s_row]).
#include "activation_quant_common.h"

static inline float64 fast_positive_reciprocal(float64 s) {
  float64 s_up = s * 16777216.0f;
  float64 r_up = v_reciprocal_fast_f32(s_up);
  float64 r = r_up * 16777216.0f;
  return v_f32_mac_b(v_f32_mac_b(-s, r, 1.0f), r, r);
}

static inline float128 fp8_quant_dequant_pair(float128 x, float64 scale2, float64 half_rcp) {
  float64 z0 = x.v1 * half_rcp;
  float64 z1 = x.v2 * half_rcp;
  z0 = v_f32_mac_b(v_f32_mac_b(-z0, scale2, x.v1), half_rcp, z0);
  z1 = v_f32_mac_b(v_f32_mac_b(-z1, scale2, x.v2), half_rcp, z1);
  z0 = v_f32_min_b(v_f32_max_b(z0, -224.0f), 224.0f);
  z1 = v_f32_min_b(v_f32_max_b(z1, -224.0f), 224.0f);
  minifloat256 q8 = (minifloat256)0;
  q8 = v_convert_f32_to_f8_b(z0, 0, SW_RHNE | SW_CLIP_FP, q8);
  q8 = v_convert_f32_to_f8_b(z1, 2, SW_RHNE | SW_CLIP_FP, q8);
  float128 w = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(q8));
  w.v1 *= scale2;
  w.v2 *= scale2;
  return w;
}

static inline void emit_one(tensor out_1k, tensor out_2k,
                            float128 x, float64 s2, float64 half_rcp,
                            int col, int row,
                            float64 inv_rscale, float64 inv_rscale256) {
  float128 w = fp8_quant_dequant_pair(x, s2, half_rcp);
  bfloat128 h = convert_float128_to_bfloat128(w, SW_RHNE);
  int5 src = {col, row, 0, 0, 0};
  v_bf16_st_tnsr(src, out_1k, h);

  float128 hw = v_convert_bf16_to_f32_all_b(h);
  float128 ax = {w.v1 * inv_rscale, w.v2 * inv_rscale};
  float128 adx = {(w.v1 - hw.v1) * inv_rscale256, (w.v2 - hw.v2) * inv_rscale256};

  int5 dst = {col * 2, row, 0, 0, 0};
  v_f8_st_tnsr(dst, out_2k, fp8_143_pack256_all_clamped(ax, adx));
}

static inline void quant_one_group(tensor input, tensor out_1k, tensor out_2k,
                                   int src_col, int row,
                                   float64 inv_rscale, float64 inv_rscale256) {
  int5 src = {src_col, row, 0, 0, 0};
  float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(src, input));
  float64 a0 = (float64)((uint64)x.v1 & 0x7fffffffu);
  float64 a1 = (float64)((uint64)x.v2 & 0x7fffffffu);
  float64 maximum = v_f32_reduce_max(v_f32_max_b(a0, a1));
  maximum = v_f32_sel_eq_f32_b(maximum, 0.0f, 1.0f, maximum);
  float64 scale = fp8_activation_scale(maximum);
  float64 s2 = scale * 2.0f;
  float64 half_rcp = fast_positive_reciprocal(s2);
  emit_one(out_1k, out_2k, x, s2, half_rcp, src_col, row, inv_rscale, inv_rscale256);
}

void main(tensor input, tensor row_scale, tensor out_1k, tensor out_2k) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(input, 0) / 128;
  int rows = get_dim_size(input, 1);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int row = r0; row < r_end; ++row) {
      int5 rp = {0, row, 0, 0, 0};
      float64 rscale = v_f32_ld_g(gen_addr(rp, row_scale));
      float64 inv_rscale = fast_positive_reciprocal(rscale);
      float64 inv_rscale256 = inv_rscale * 256.0f;

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
        mp = v_f32_sel_eq_f32_b(mp, 0.0f, 1.0f, mp);
        float64 sp2 = fp8_activation_scale(mp) * 2.0f;
        float64 hrcpp = fast_positive_reciprocal(sp2);

        float64 s0 = v_f32_mov_dual_group_all_b(sp2, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), sp2);
        float64 r0v = v_f32_mov_dual_group_all_b(hrcpp, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), hrcpp);
        emit_one(out_1k, out_2k, x0, s0, r0v, c0, row, inv_rscale, inv_rscale256);

        float64 s1 = v_f32_mov_dual_group_all_b(sp2, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), sp2);
        float64 r1v = v_f32_mov_dual_group_all_b(hrcpp, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), hrcpp);
        emit_one(out_1k, out_2k, x1, s1, r1v, c1, row, inv_rscale, inv_rscale256);
      }
      if (g < g_end) {
        quant_one_group(input, out_1k, out_2k, g * 128, row, inv_rscale, inv_rscale256);
      }
    }
  }
}
