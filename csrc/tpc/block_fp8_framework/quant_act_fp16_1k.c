// SPDX-License-Identifier: Apache-2.0
// Dedicated 1-output group-128 dynamic OCP E4M3FN activation producer for SRAM-bundled
// 1K FP16 + 2K FP8_143 GEMM (FP16 x0 when RESIDUAL_FP8_2K is unset,
// FP8_143 [xs / 128, 32 * (xs - x0)] when RESIDUAL_FP8_2K is set).
#include "activation_quant_common.h"

void main(tensor input, tensor row_scale, tensor output, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(input, 0) / 128;
  int rows = get_dim_size(input, 1);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int row = r0; row < r_end; ++row) {
      int5 rp = {0, row, 0, 0, 0};
      float rscale = s_f32_ld_g(gen_addr(rp, row_scale));
      float inv_rscale = as_float(0x7f000000u - as_uint(rscale));
      int g0 = begin[0] * 4;
      int g_end = s_i32_min(end[0] * 4, groups);
      #pragma clang loop unroll(disable)
      for (int g = g0; g < g_end; ++g) {
        int col = g * 128;
        int5 pos = {col, row, 0, 0, 0};
        float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pos, input));
        float64 maximum = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x.v1 & 0x7fffffffu),
                                                       (float64)((uint64)x.v2 & 0x7fffffffu)));
        // Substitute 1.0f denominator ONLY when a 128-element group is identically zero.
        maximum = v_f32_sel_eq_f32_b(maximum, 0.0f, 1.0f, maximum);
        float64 scale = fp8_activation_scale(maximum);
        float64 rcp = fp8_exact_positive_rcp(scale);
        float128 xs = {fp8_ocp_dequant_vec(x.v1, scale, rcp) * inv_rscale,
                       fp8_ocp_dequant_vec(x.v2, scale, rcp) * inv_rscale};
        half128 h = convert_float128_to_half128(xs, SW_RHNE);
#ifdef RESIDUAL_FP8_2K
        float128 x0 = v_convert_f16_to_f32_all_b(h);
        float128 ax = {xs.v1 * (1.0f / 128.0f), xs.v2 * (1.0f / 128.0f)};
        float128 adx = {(xs.v1 - x0.v1) * 32.0f, (xs.v2 - x0.v2) * 32.0f};
        int5 dst = {col * 2, row, 0, 0, 0};
        v_f8_st_tnsr(dst, output, fp8_143_pack256_all_clamped(ax, adx));
#else
        int5 dst = {col, row, 0, 0, 0};
        v_f16_st_tnsr(dst, output, h);
#endif
      }
    }
  }
}
