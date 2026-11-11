// SPDX-License-Identifier: Apache-2.0
#include "activation_quant_common.h"

// Exact OCP E4M3FN round-to-nearest-even in native BF16 even/odd lane order,
// preserving the full 2^-9 subnormal grid losslessly in BF16 (7-bit mantissa >=
// 3-bit E4M3FN mantissa) along with row-normalized group scale sc / row_scale.
static inline float64 fp8_ocp_quant_val_vec(float64 x, float64 scale, float64 reciprocal) {
  float64 q = x * reciprocal;
  float64 residual = v_f32_mac_b(-q, scale, x);
  float64 z = v_f32_mac_b(residual, reciprocal, q);
  float64 magnitude = (float64)((uint64)z & 0x7fffffffu);
  magnitude = v_f32_min_b(magnitude, 448.0f);
  uint64 bits = (uint64)magnitude;
  uint64 rounded = (bits + 0x7ffffu + ((bits >> 20) & 1u)) & 0xfff00000u;
  float64 normal = (float64)rounded;
  float64 subnormal = v_f32_sub_b(v_f32_add_b(magnitude, 16384.0f), 16384.0f);
  float64 value = v_f32_sel_less_f32_b(magnitude, 0.015625f, subnormal, normal);
  return (float64)((uint64)value | ((uint64)x & 0x80000000u));
}

static inline void prequant_group_one(tensor input, tensor q_out, tensor s_norm, int g, int row,
                                      float64 inv_rscale) {
  int5 pos = {g * 128, row, 0, 0, 0};
  float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pos, input));
  float64 raw_max = v_f32_reduce_max(v_f32_max_b((float64)((uint64)x.v1 & 0x7fffffffu),
                                                 (float64)((uint64)x.v2 & 0x7fffffffu)));
  float64 vmax = v_f32_sel_eq_f32_b(raw_max, 0.0f, 1.0f, raw_max);
  float64 sc = fp8_activation_scale(vmax);
  float64 rcp = fp8_exact_positive_rcp(sc);
  float128 q = {fp8_ocp_quant_val_vec(x.v1, sc, rcp), fp8_ocp_quant_val_vec(x.v2, sc, rcp)};
  float64 sn = v_f32_sel_eq_f32_b(raw_max, 0.0f, 0.0f, sc * inv_rscale);
  int5 q_pos = {0, row, g, 0, 0};
  v_bf16_st_tnsr(q_pos, q_out, convert_float128_to_bfloat128(q, SW_RHNE));
  v_f32_st_tnsr_partial(q_pos, s_norm, sn, 0, 0);
}

void main(tensor input, tensor row_scale, tensor q_out, tensor s_norm, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(input, 0) / 128;
  int rows = get_dim_size(input, 1);
  int pairs = (groups + 1) / 2;
  int gp0 = begin[0];
  int gp_end = s_i32_min(end[0], pairs);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 16;
    int r_end = s_i32_min(r0 + 16, rows);
    for (int gp = gp0; gp < gp_end; ++gp) {
      int g0 = gp * 2;
      if (g0 + 1 < groups) {
        int row = r0;
        for (; row + 1 < r_end; row += 2) {
          int5 rp_a = {0, row, 0, 0, 0}, rp_b = {0, row + 1, 0, 0, 0};
          float64 inv_a = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp_a, row_scale)));
          float64 inv_b = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp_b, row_scale)));
          int5 p0a = {g0 * 128, row, 0, 0, 0}, p1a = {(g0 + 1) * 128, row, 0, 0, 0};
          int5 p0b = {g0 * 128, row + 1, 0, 0, 0}, p1b = {(g0 + 1) * 128, row + 1, 0, 0, 0};
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

          float128 q0a = {fp8_ocp_quant_val_vec(x0a.v1, sc0a, rcp0a), fp8_ocp_quant_val_vec(x0a.v2, sc0a, rcp0a)};
          float128 q1a = {fp8_ocp_quant_val_vec(x1a.v1, sc1a, rcp1a), fp8_ocp_quant_val_vec(x1a.v2, sc1a, rcp1a)};
          float128 q0b = {fp8_ocp_quant_val_vec(x0b.v1, sc0b, rcp0b), fp8_ocp_quant_val_vec(x0b.v2, sc0b, rcp0b)};
          float128 q1b = {fp8_ocp_quant_val_vec(x1b.v1, sc1b, rcp1b), fp8_ocp_quant_val_vec(x1b.v2, sc1b, rcp1b)};

          int5 qp0a = {0, row, g0, 0, 0}, qp1a = {0, row, g0 + 1, 0, 0};
          int5 qp0b = {0, row + 1, g0, 0, 0}, qp1b = {0, row + 1, g0 + 1, 0, 0};
          v_bf16_st_tnsr(qp0a, q_out, convert_float128_to_bfloat128(q0a, SW_RHNE));
          v_bf16_st_tnsr(qp1a, q_out, convert_float128_to_bfloat128(q1a, SW_RHNE));
          v_bf16_st_tnsr(qp0b, q_out, convert_float128_to_bfloat128(q0b, SW_RHNE));
          v_bf16_st_tnsr(qp1b, q_out, convert_float128_to_bfloat128(q1b, SW_RHNE));

          v_f32_st_tnsr_partial(qp0a, s_norm, v_f32_sel_eq_f32_b(rm0a, 0.0f, 0.0f, sc0a * inv_a), 0, 0);
          v_f32_st_tnsr_partial(qp1a, s_norm, v_f32_sel_eq_f32_b(rm1a, 0.0f, 0.0f, sc1a * inv_a), 0, 0);
          v_f32_st_tnsr_partial(qp0b, s_norm, v_f32_sel_eq_f32_b(rm0b, 0.0f, 0.0f, sc0b * inv_b), 0, 0);
          v_f32_st_tnsr_partial(qp1b, s_norm, v_f32_sel_eq_f32_b(rm1b, 0.0f, 0.0f, sc1b * inv_b), 0, 0);
        }
        for (; row < r_end; ++row) {
          int5 rp = {0, row, 0, 0, 0};
          float64 inv_rs = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp, row_scale)));
          prequant_group_one(input, q_out, s_norm, g0, row, inv_rs);
          prequant_group_one(input, q_out, s_norm, g0 + 1, row, inv_rs);
        }
      } else {
        for (int row = r0; row < r_end; ++row) {
          int5 rp = {0, row, 0, 0, 0};
          float64 inv_rs = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(rp, row_scale)));
          prequant_group_one(input, q_out, s_norm, g0, row, inv_rs);
        }
      }
    }
  }
}
