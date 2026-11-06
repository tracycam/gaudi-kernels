// SPDX-License-Identifier: Apache-2.0
// FCD-first scores/probabilities: [KV tile, query tile, heads]. Header contains
// valid query rows, valid key rows, and the tile's first logical key position.
// The BF16 output is exp(score * scale - tile_max), before normalization.
#include "../common/tpc_math_safe.h"

void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, tensor tile_max, tensor tile_sum, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 h0 = {0, 0, 0, 0, 0}, h1 = {1, 0, 0, 0, 0}, h2 = {2, 0, 0, 0, 0};
  int valid_rows = s_i32_ld_g(gen_addr(h0, header));
  int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  int key_start = s_i32_ld_g(gen_addr(h2, header));
  int keys = get_dim_size(scores, 0);
  uint64 lane = read_lane_id_4b_b();
  float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int row = begin[1]; row < end[1]; ++row) {
      int5 pp = {row, 0, 0, 0, 0};
      int position = s_i32_ld_g(gen_addr(pp, positions));
      int count = row < valid_rows ? s_i32_max(0, s_i32_min(valid_keys, position - key_start + 1)) : 0;
      float64 max0 = ni, max1 = ni;
      for (int k = 0; k < keys; k += 128) {
        int lo = s_i32_max(0, s_i32_min(64, count - k));
        int hi = s_i32_max(0, s_i32_min(64, count - k - 64));
        int5 p = {k, row, head, 0, 0};
        float64 a = v_f32_ld_tnsr_b(p, scores) * scale;
        p[0] += 64;
        float64 b = v_f32_ld_tnsr_b(p, scores) * scale;
        max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b(lane, lo, a, ni));
        max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b(lane, hi, b, ni));
      }
      float64 maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
      float64 origin = v_f32_sel_eq_f32_b(maximum, ni, 0.f, maximum);
      float64 sum0 = 0.f, sum1 = 0.f;
      for (int k = 0; k < keys; k += 128) {
        int lo = s_i32_max(0, s_i32_min(64, count - k));
        int hi = s_i32_max(0, s_i32_min(64, count - k - 64));
        int5 p = {k, row, head, 0, 0};
        float64 a = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - origin);
        p[0] += 64;
        float64 b = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - origin);
        a = v_f32_sel_less_u32_b(lane, lo, a, 0.f);
        b = v_f32_sel_less_u32_b(lane, hi, b, 0.f);
        sum0 += a;
        sum1 += b;
        float128 pair = {a, b};
        p[0] -= 64;
        v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
      }
      float64 sum = v_f32_reduce_add(sum0 + sum1);
      int5 stat = {row, head, 0, 0, 0};
      v_f32_st_tnsr_partial(stat, tile_max, maximum, 0, 0);
      v_f32_st_tnsr_partial(stat, tile_sum, sum, 0, 0);
    }
  }
}
