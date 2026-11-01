// SPDX-License-Identifier: Apache-2.0
// M=32 causal FP32 softmax split into BF16 high and residual MME parts.
#include "../common/tpc_math_safe.h"

static inline int valid_count(tensor pages, tensor usage, tensor groups,
                              int page, int physical_pages, int position) {
  int5 p = {page, 0, 0, 0, 0};
  int physical = s_i32_ld_g(gen_addr(p, pages));
  int count = s_i32_ld_g(gen_addr(p, usage));
  int group = s_i32_ld_g(gen_addr(p, groups));
  int causal = s_i32_max(0, s_i32_min(position + 1 - page * 128, 128));
  return group == 0 && physical >= 0 && physical < physical_pages
             ? s_i32_max(0, s_i32_min(s_i32_min(count, 128), causal)) : 0;
}

void main(tensor scores, tensor pages, tensor usage, tensor groups, tensor positions,
          tensor probabilities, tensor corrections, float scale, int physical_pages) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int page_count = get_dim_size(pages, 0);
  uint64 lane = read_lane_id_4b_b();
  float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    int5 rp = {head / 16, 0, 0, 0, 0};
    int position = s_i32_ld_g(gen_addr(rp, positions));
    float64 max0 = ni, max1 = ni;
    for (int page = 0; page < page_count; ++page) {
      int count = valid_count(pages, usage, groups, page, physical_pages, position);
      unsigned lo = s_i32_min(count, 64), hi = s_i32_max(0, count - 64);
      int5 p = {page * 128, head, 0, 0, 0};
      float64 a = v_f32_ld_tnsr_b(p, scores) * scale;
      p[0] += 64;
      float64 b = v_f32_ld_tnsr_b(p, scores) * scale;
      max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b(lane, lo, a, ni));
      max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b(lane, hi, b, ni));
    }
    float64 maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
    float64 sum0 = 0, sum1 = 0;
    for (int page = 0; page < page_count; ++page) {
      int count = valid_count(pages, usage, groups, page, physical_pages, position);
      unsigned lo = s_i32_min(count, 64), hi = s_i32_max(0, count - 64);
      int5 p = {page * 128, head, 0, 0, 0};
      float64 a = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - maximum);
      p[0] += 64;
      float64 b = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - maximum);
      sum0 += v_f32_sel_less_u32_b(lane, lo, a, 0.f);
      sum1 += v_f32_sel_less_u32_b(lane, hi, b, 0.f);
    }
    float64 reciprocal = v_reciprocal_f32(v_f32_reduce_add(sum0 + sum1));
    for (int page = 0; page < page_count; ++page) {
      int count = valid_count(pages, usage, groups, page, physical_pages, position);
      unsigned lo = s_i32_min(count, 64), hi = s_i32_max(0, count - 64);
      int5 p = {page * 128, head, 0, 0, 0};
      float64 a = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - maximum) * reciprocal;
      p[0] += 64;
      float64 b = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p, scores) * scale - maximum) * reciprocal;
      float128 pair = {v_f32_sel_less_u32_b(lane, lo, a, 0.f),
                       v_f32_sel_less_u32_b(lane, hi, b, 0.f)};
      bfloat128 high = convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE);
      float128 reconstructed = convert_bfloat128_to_float128(high, SW_LINEAR);
      float128 low = {pair.v1 - reconstructed.v1, pair.v2 - reconstructed.v2};
      p[0] -= 64;
      v_bf16_st_tnsr(p, probabilities, high);
      v_bf16_st_tnsr(p, corrections, convert_float128_to_bfloat128(low, SW_LINEAR | SW_RHNE));
    }
  }
}
