// SPDX-License-Identifier: Apache-2.0
// Fused SRAM-bundled activation producer for 1K FP16 block-FP8 prefill:
// quantizes each 128-element BF16 group to exact OCP E4M3FN in registers and
// emits row-normalized FP16 (q * (group_scale / row_scale)) directly to SRAM.
#include "activation_quant_common.h"

static inline void quant_group_one(tensor input, tensor output, int col_offset, int g, int row,
                                   float64 inv_rscale) {
  int5 src = {col_offset + g * 128, row, 0, 0, 0};
  float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(src, input));
  float64 raw_max = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x.v1 & 0x7fffffffu),
                                                 (float64)((uint64)x.v2 & 0x7fffffffu)));
  float64 vmax = v_f32_sel_eq_f32_b(raw_max, 0.0f, 1.0f, raw_max);
  float64 sc = fp8_activation_scale(vmax);
  float64 rcp = fp8_exact_positive_rcp(sc);
  float64 sn = v_f32_sel_eq_f32_b(raw_max, 0.0f, 0.0f, sc * inv_rscale);
  float128 xs = {fp8_ocp_quant_val_vec(x.v1, sc, rcp) * sn,
                 fp8_ocp_quant_val_vec(x.v2, sc, rcp) * sn};
  int5 dst = {g * 128, row, 0, 0, 0};
  v_f16_st_tnsr(dst, output, convert_float128_to_half128(xs, SW_RHNE));
}

void main(tensor input, tensor row_scale, tensor output, int col_offset, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(output, 0) / 128;
  int rows = get_dim_size(output, 1);
  int g_start = begin[0] * 4;
  int g_end = s_i32_min(end[0] * 4, groups);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 16;
    int r_end = s_i32_min(r0 + 16, rows);
    int g0 = g_start;
    for (; g0 + 1 < g_end; g0 += 2) {
      int row = r0;
      for (; row + 1 < r_end; row += 2) {
        int5 rp_a = {0, row, 0, 0, 0}, rp_b = {0, row + 1, 0, 0, 0};
        float64 inv_a = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp_a, row_scale)));
        float64 inv_b = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp_b, row_scale)));
        int c0 = col_offset + g0 * 128, c1 = col_offset + (g0 + 1) * 128;
        int5 p0a = {c0, row, 0, 0, 0}, p1a = {c1, row, 0, 0, 0};
        int5 p0b = {c0, row + 1, 0, 0, 0}, p1b = {c1, row + 1, 0, 0, 0};
        bfloat128 bx0a = v_bf16_ld_tnsr_b(p0a, input), bx1a = v_bf16_ld_tnsr_b(p1a, input);
        bfloat128 bx0b = v_bf16_ld_tnsr_b(p0b, input), bx1b = v_bf16_ld_tnsr_b(p1b, input);
        float128 x0a = v_convert_bf16_to_f32_all_b(bx0a), x1a = v_convert_bf16_to_f32_all_b(bx1a);
        float128 x0b = v_convert_bf16_to_f32_all_b(bx0b), x1b = v_convert_bf16_to_f32_all_b(bx1b);

        float64 rm0a = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x0a.v1 & 0x7fffffffu),
                                                    (float64)((uint64)x0a.v2 & 0x7fffffffu)));
        float64 rm1a = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x1a.v1 & 0x7fffffffu),
                                                    (float64)((uint64)x1a.v2 & 0x7fffffffu)));
        float64 rm0b = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x0b.v1 & 0x7fffffffu),
                                                    (float64)((uint64)x0b.v2 & 0x7fffffffu)));
        float64 rm1b = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x1b.v1 & 0x7fffffffu),
                                                    (float64)((uint64)x1b.v2 & 0x7fffffffu)));

        float64 sc0a = fp8_activation_scale(v_f32_sel_eq_f32_b(rm0a, 0.0f, 1.0f, rm0a));
        float64 sc1a = fp8_activation_scale(v_f32_sel_eq_f32_b(rm1a, 0.0f, 1.0f, rm1a));
        float64 sc0b = fp8_activation_scale(v_f32_sel_eq_f32_b(rm0b, 0.0f, 1.0f, rm0b));
        float64 sc1b = fp8_activation_scale(v_f32_sel_eq_f32_b(rm1b, 0.0f, 1.0f, rm1b));
        float64 rcp0a = fp8_exact_positive_rcp(sc0a), rcp1a = fp8_exact_positive_rcp(sc1a);
        float64 rcp0b = fp8_exact_positive_rcp(sc0b), rcp1b = fp8_exact_positive_rcp(sc1b);

        float64 sn0a = v_f32_sel_eq_f32_b(rm0a, 0.0f, 0.0f, sc0a * inv_a);
        float64 sn1a = v_f32_sel_eq_f32_b(rm1a, 0.0f, 0.0f, sc1a * inv_a);
        float64 sn0b = v_f32_sel_eq_f32_b(rm0b, 0.0f, 0.0f, sc0b * inv_b);
        float64 sn1b = v_f32_sel_eq_f32_b(rm1b, 0.0f, 0.0f, sc1b * inv_b);

        float128 xs0a = {fp8_ocp_quant_val_vec(x0a.v1, sc0a, rcp0a) * sn0a,
                         fp8_ocp_quant_val_vec(x0a.v2, sc0a, rcp0a) * sn0a};
        float128 xs1a = {fp8_ocp_quant_val_vec(x1a.v1, sc1a, rcp1a) * sn1a,
                         fp8_ocp_quant_val_vec(x1a.v2, sc1a, rcp1a) * sn1a};
        float128 xs0b = {fp8_ocp_quant_val_vec(x0b.v1, sc0b, rcp0b) * sn0b,
                         fp8_ocp_quant_val_vec(x0b.v2, sc0b, rcp0b) * sn0b};
        float128 xs1b = {fp8_ocp_quant_val_vec(x1b.v1, sc1b, rcp1b) * sn1b,
                         fp8_ocp_quant_val_vec(x1b.v2, sc1b, rcp1b) * sn1b};

        int d0 = g0 * 128, d1 = (g0 + 1) * 128;
        int5 dp0a = {d0, row, 0, 0, 0}, dp1a = {d1, row, 0, 0, 0};
        int5 dp0b = {d0, row + 1, 0, 0, 0}, dp1b = {d1, row + 1, 0, 0, 0};
        v_f16_st_tnsr(dp0a, output, convert_float128_to_half128(xs0a, SW_RHNE));
        v_f16_st_tnsr(dp1a, output, convert_float128_to_half128(xs1a, SW_RHNE));
        v_f16_st_tnsr(dp0b, output, convert_float128_to_half128(xs0b, SW_RHNE));
        v_f16_st_tnsr(dp1b, output, convert_float128_to_half128(xs1b, SW_RHNE));
      }
      for (; row < r_end; ++row) {
        int5 rp = {0, row, 0, 0, 0};
        float64 inv_rs = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp, row_scale)));
        quant_group_one(input, output, col_offset, g0, row, inv_rs);
        quant_group_one(input, output, col_offset, g0 + 1, row, inv_rs);
      }
    }
    for (; g0 < g_end; ++g0) {
      for (int row = r0; row < r_end; ++row) {
        int5 rp = {0, row, 0, 0, 0};
        float64 inv_rs = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp, row_scale)));
        quant_group_one(input, output, col_offset, g0, row, inv_rs);
      }
    }
  }
}
