// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Merge the per-tile MME results in the same order as paged prefill, keeping
// the accumulator in registers and producing the final BF16 output once.
void main(tensor tile_max, tensor tile_sum, tensor tile_output, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(tile_output, 0), tiles = get_dim_size(tile_output, 1);
  const float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    float64 m0 = 0.f, l0 = 0.f, x0 = 0.f, x1 = 0.f;
    for (int tile = 0; tile < tiles; ++tile) {
      int5 stat = {0,tile,head,0,0};
      float64 m1 = v_f32_ld_tnsr_b(stat, tile_max), l1 = v_f32_ld_tnsr_b(stat, tile_sum);
      float64 valid0 = v_f32_sel_grt_f32_b(l0, 0.f, m0, (float64)ni);
      float64 valid1 = v_f32_sel_grt_f32_b(l1, 0.f, m1, (float64)ni);
      float64 m = v_f32_max_b(valid0, valid1);
      float64 origin = v_f32_sel_eq_f32_b(m, ni, 0.f, m);
      float64 arg0 = v_f32_sel_grt_f32_b(l0, 0.f, m0 - origin, 0.f);
      float64 arg1 = v_f32_sel_grt_f32_b(l1, 0.f, m1 - origin, 0.f);
      float64 a = v_f32_sel_grt_f32_b(l0, 0.f, tpc_exp_f32_safe(arg0), 0.f);
      float64 b = v_f32_sel_grt_f32_b(l1, 0.f, tpc_exp_f32_safe(arg1), 0.f);
      float64 l = a * l0 + b * l1;
      float64 aa = v_broadcast_element_f32(a, 0), bb = v_broadcast_element_f32(b * l1, 0);
      int5 p = {0,tile,head,0,0};
      x0 = aa * x0 + bb * v_f32_ld_tnsr_b(p, tile_output);
      if (width == 128) {
        p[0] = 64;
        x1 = aa * x1 + bb * v_f32_ld_tnsr_b(p, tile_output);
      }
      m0 = m;
      l0 = l;
    }
    float64 safe = v_f32_sel_grt_f32_b(l0, 0.f, l0, 1.f);
    float64 inverse = v_f32_sel_grt_f32_b(l0, 0.f, v_reciprocal_f32(safe), 0.f);
    float64 r = v_broadcast_element_f32(inverse, 0);
    float128 pair = {x0 * r, x1 * r};
    bfloat128 result = convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE);
    int5 out = {0,head,0,0,0};
    if (width == 128) v_bf16_st_tnsr(out, output, result);
    else v_bf16_st_tnsr_partial(out, output, result, 63, 0);
  }
}
