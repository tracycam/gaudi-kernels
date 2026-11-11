// SPDX-License-Identifier: Apache-2.0
#include "activation_quant_common.h"

// Emit SRAM-bundled activation operands from pre-quantized (q_in, s_norm):
// - default (1K FP16): x0 = FP16(q * s_norm) in all_b lane order;
// - RESIDUAL_FP8_3K (3K FP8_143): 2-stage coarse+fine x0 & FP16 residual (dx)
//   packed 256 bytes per group-pair so x0*dw + dx*w0 is reconstructed within 1e-4.
void main(tensor q_in, tensor s_norm, tensor output, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int rows = get_dim_size(q_in, 1);
  int groups = get_dim_size(q_in, 2);
#ifdef RESIDUAL_FP8_3K
  int pairs = (groups + 1) / 2;
  int gp0 = begin[0] * 2;
  int gp_end = s_i32_min(end[0] * 2, pairs);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 16;
    int r_end = s_i32_min(r0 + 16, rows);
    for (int gp = gp0; gp < gp_end; ++gp) {
      int g0 = gp * 2;
      int base = gp * 768;
      if (g0 + 1 < groups) {
        #pragma clang loop unroll_count(2)
        for (int row = r0; row < r_end; ++row) {
          int5 p0 = {0, row, g0, 0, 0}, p1 = {0, row, g0 + 1, 0, 0};
          float64 sn0 = v_f32_ld_g(gen_addr(p0, s_norm));
          float64 sn1 = v_f32_ld_g(gen_addr(p1, s_norm));
          float128 q0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0, q_in));
          float128 q1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1, q_in));
          float128 xs0 = {q0.v1 * sn0, q0.v2 * sn0};
          float128 xs1 = {q1.v1 * sn1, q1.v2 * sn1};
          float128 x0_0 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(xs0, SW_RHNE));
          float128 x0_1 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(xs1, SW_RHNE));
          float256 xs_s = {xs0.v1 * (1.0f / 128.0f), xs0.v2 * (1.0f / 128.0f),
                           xs1.v1 * (1.0f / 128.0f), xs1.v2 * (1.0f / 128.0f)};
          minifloat256 v0 = v_convert_f32_to_f8_all_b(xs_s, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          float256 ax0 = v_convert_f8_to_f32_all_b(v0);
          float256 ax1 = {(xs_s.v1 - ax0.v1) * 8.0f, (xs_s.v2 - ax0.v2) * 8.0f,
                          (xs_s.v3 - ax0.v3) * 8.0f, (xs_s.v4 - ax0.v4) * 8.0f};
          float256 adx = {(xs0.v1 - x0_0.v1) * 32.0f, (xs0.v2 - x0_0.v2) * 32.0f,
                          (xs1.v1 - x0_1.v1) * 32.0f, (xs1.v2 - x0_1.v2) * 32.0f};
          minifloat256 v1 = v_convert_f32_to_f8_all_b(ax1, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          minifloat256 v2 = v_convert_f32_to_f8_all_b(adx, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          int5 d0 = {base, row, 0, 0, 0}, d1 = {base + 256, row, 0, 0, 0}, d2 = {base + 512, row, 0, 0, 0};
          v_f8_st_tnsr(d0, output, v0);
          v_f8_st_tnsr(d1, output, v1);
          v_f8_st_tnsr(d2, output, v2);
        }
      } else {
        #pragma clang loop unroll_count(2)
        for (int row = r0; row < r_end; ++row) {
          int5 p0 = {0, row, g0, 0, 0};
          float64 sn0 = v_f32_ld_g(gen_addr(p0, s_norm));
          float128 q0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0, q_in));
          float128 xs0 = {q0.v1 * sn0, q0.v2 * sn0};
          float128 x0_0 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(xs0, SW_RHNE));
          float256 xs_s = {xs0.v1 * (1.0f / 128.0f), xs0.v2 * (1.0f / 128.0f), 0.0f, 0.0f};
          minifloat256 v0 = v_convert_f32_to_f8_all_b(xs_s, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          float256 ax0 = v_convert_f8_to_f32_all_b(v0);
          float256 ax1 = {(xs_s.v1 - ax0.v1) * 8.0f, (xs_s.v2 - ax0.v2) * 8.0f, 0.0f, 0.0f};
          float256 adx = {(xs0.v1 - x0_0.v1) * 32.0f, (xs0.v2 - x0_0.v2) * 32.0f, 0.0f, 0.0f};
          minifloat256 v1 = v_convert_f32_to_f8_all_b(ax1, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          minifloat256 v2 = v_convert_f32_to_f8_all_b(adx, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
          int5 d0 = {base, row, 0, 0, 0}, d1 = {base + 256, row, 0, 0, 0}, d2 = {base + 512, row, 0, 0, 0};
          v_f8_st_tnsr(d0, output, v0);
          v_f8_st_tnsr(d1, output, v1);
          v_f8_st_tnsr(d2, output, v2);
        }
      }
    }
  }
#else
  int g0 = begin[0] * 4;
  int g_end = s_i32_min(end[0] * 4, groups);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 16;
    int r_end = s_i32_min(r0 + 16, rows);
    for (int g = g0; g < g_end; ++g) {
      int col = g * 128;
      #pragma clang loop unroll_count(4)
      for (int row = r0; row < r_end; ++row) {
        int5 p = {0, row, g, 0, 0};
        float64 sn = v_f32_ld_g(gen_addr(p, s_norm));
        float128 q = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, q_in));
        float128 xs = {q.v1 * sn, q.v2 * sn};
        int5 dst = {col, row, 0, 0, 0};
        v_f16_st_tnsr(dst, output, convert_float128_to_half128(xs, SW_RHNE));
      }
    }
  }
#endif
}
