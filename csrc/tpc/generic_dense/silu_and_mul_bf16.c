// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

static inline float64 silu_half_f32(float64 g) {
  float64 denom = 1.0f + tpc_exp_f32_safe(-g);
  float64 inv = v_reciprocal_fast_f32(denom);
  float64 q = g * inv;
  float64 res = v_f32_mac_b(-q, denom, g);
  float64 refined = v_f32_mac_b(res, inv, q);
  float64 val = v_f32_sel_leq_f32_b(g, -88.0f, 0.0f, refined);
  uint64 gabs = (uint64)g & 0x7fffffffu;
  return (float64)v_u32_sel_grt_u32_b(gabs, 0x7f800000u, (uint64)g, (uint64)val);
}

// The fused primitive rounds only its final product to BF16.
void main(tensor gate_up, tensor output) {
  const int width = get_dim_size(output, 0);
  const int rows = get_dim_size(output, 1);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int cb = begin[0]; cb < end[0]; ++cb) {
      int c0 = cb * 512;
      int c_end = s_i32_min(c0 + 512, width);
      for (int row = r0; row < r_end; ++row) {
        int col = c0;
        for (; col + 256 <= c_end; col += 256) {
          int5 gp0 = {col, row, 0, 0, 0}, up0 = {col + width, row, 0, 0, 0};
          int5 gp1 = {col + 128, row, 0, 0, 0}, up1 = {col + 128 + width, row, 0, 0, 0};
          float128 g0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gp0, gate_up));
          float128 g1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gp1, gate_up));

          float64 d0a = 1.0f + tpc_exp_f32_safe(-g0.v1);
          float64 d0b = 1.0f + tpc_exp_f32_safe(-g0.v2);
          float64 d1a = 1.0f + tpc_exp_f32_safe(-g1.v1);
          float64 d1b = 1.0f + tpc_exp_f32_safe(-g1.v2);

          float64 i0a = v_reciprocal_fast_f32(d0a);
          float64 i0b = v_reciprocal_fast_f32(d0b);
          float64 i1a = v_reciprocal_fast_f32(d1a);
          float64 i1b = v_reciprocal_fast_f32(d1b);

          float128 u0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(up0, gate_up));
          float128 u1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(up1, gate_up));

          float64 q0a = g0.v1 * i0a, q0b = g0.v2 * i0b;
          float64 q1a = g1.v1 * i1a, q1b = g1.v2 * i1b;

          q0a = v_f32_mac_b(v_f32_mac_b(-q0a, d0a, g0.v1), i0a, q0a);
          q0b = v_f32_mac_b(v_f32_mac_b(-q0b, d0b, g0.v2), i0b, q0b);
          q1a = v_f32_mac_b(v_f32_mac_b(-q1a, d1a, g1.v1), i1a, q1a);
          q1b = v_f32_mac_b(v_f32_mac_b(-q1b, d1b, g1.v2), i1b, q1b);

          q0a = v_f32_sel_leq_f32_b(g0.v1, -88.0f, 0.0f, q0a);
          q0b = v_f32_sel_leq_f32_b(g0.v2, -88.0f, 0.0f, q0b);
          q1a = v_f32_sel_leq_f32_b(g1.v1, -88.0f, 0.0f, q1a);
          q1b = v_f32_sel_leq_f32_b(g1.v2, -88.0f, 0.0f, q1b);

          q0a = (float64)v_u32_sel_grt_u32_b((uint64)g0.v1 & 0x7fffffffu, 0x7f800000u, (uint64)g0.v1, (uint64)q0a);
          q0b = (float64)v_u32_sel_grt_u32_b((uint64)g0.v2 & 0x7fffffffu, 0x7f800000u, (uint64)g0.v2, (uint64)q0b);
          q1a = (float64)v_u32_sel_grt_u32_b((uint64)g1.v1 & 0x7fffffffu, 0x7f800000u, (uint64)g1.v1, (uint64)q1a);
          q1b = (float64)v_u32_sel_grt_u32_b((uint64)g1.v2 & 0x7fffffffu, 0x7f800000u, (uint64)g1.v2, (uint64)q1b);

          float128 p0 = {q0a * u0.v1, q0b * u0.v2};
          float128 p1 = {q1a * u1.v1, q1b * u1.v2};
          v_bf16_st_tnsr(gp0, output, convert_float128_to_bfloat128(p0, SW_RHNE));
          v_bf16_st_tnsr(gp1, output, convert_float128_to_bfloat128(p1, SW_RHNE));
        }
        for (; col < c_end; col += 128) {
          int5 gate_pos = {col, row, 0, 0, 0};
          int5 up_pos = {col + width, row, 0, 0, 0};
          if (c_end - col < 128) {
            int count = c_end - col - 1;
            float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(gate_pos, gate_up, (char)count, 0));
            float128 up = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(up_pos, gate_up, (char)count, 0));
            float128 product = {silu_half_f32(gate.v1) * up.v1, silu_half_f32(gate.v2) * up.v2};
            v_bf16_st_tnsr_partial(gate_pos, output, convert_float128_to_bfloat128(product, SW_RHNE), (char)count, 0);
          } else {
            float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gate_pos, gate_up));
            float128 up = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(up_pos, gate_up));
            float128 product = {silu_half_f32(gate.v1) * up.v1, silu_half_f32(gate.v2) * up.v2};
            v_bf16_st_tnsr(gate_pos, output, convert_float128_to_bfloat128(product, SW_RHNE));
          }
        }
      }
    }
  }
}
