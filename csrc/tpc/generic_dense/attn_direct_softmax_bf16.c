// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Single-block causal softmax keeps up to 512 scaled scores in vector registers.
void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 h0 = {0,0,0,0,0}, h1 = {1,0,0,0,0}, h2 = {2,0,0,0,0};
  const int valid_rows = s_i32_ld_g(gen_addr(h0, header));
  const int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  const int key_start = s_i32_ld_g(gen_addr(h2, header));
  const int keys = get_dim_size(scores, 0);
  const uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int row = begin[1]; row < end[1]; ++row) {
      const int item = row & 63;
      int5 pos = {row - item,0,0,0,0};
      int64 pos_vector = v_i32_ld_tnsr_b(pos, positions);
        float64 maximum = ni, sum = 0.f;
        int5 p = {0,row,head,0,0};
        if (row < valid_rows) {
          int64 position = v_broadcast_element_i32(pos_vector, item);
          int64 last_key = v_i32_min_b(position - key_start, (int64)(valid_keys - 1));
          float64 max0 = ni, max1 = ni;

          float64 score0 = 0.f;
          if (keys > 0) {
            p[0] = 0; score0 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 0, score0, ni);
            max0 = v_f32_max_b(max0, m);
          }
          float64 score1 = 0.f;
          if (keys > 64) {
            p[0] = 64; score1 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 64, score1, ni);
            max1 = v_f32_max_b(max1, m);
          }
          float64 score2 = 0.f;
          if (keys > 128) {
            p[0] = 128; score2 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 128, score2, ni);
            max0 = v_f32_max_b(max0, m);
          }
          float64 score3 = 0.f;
          if (keys > 192) {
            p[0] = 192; score3 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 192, score3, ni);
            max1 = v_f32_max_b(max1, m);
          }
          float64 score4 = 0.f;
          if (keys > 256) {
            p[0] = 256; score4 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 256, score4, ni);
            max0 = v_f32_max_b(max0, m);
          }
          float64 score5 = 0.f;
          if (keys > 320) {
            p[0] = 320; score5 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 320, score5, ni);
            max1 = v_f32_max_b(max1, m);
          }
          float64 score6 = 0.f;
          if (keys > 384) {
            p[0] = 384; score6 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 384, score6, ni);
            max0 = v_f32_max_b(max0, m);
          }
          float64 score7 = 0.f;
          if (keys > 448) {
            p[0] = 448; score7 = v_f32_ld_tnsr_b(p, scores) * scale;
            float64 m = v_f32_sel_leq_i32_b((int64)lanes, last_key - 448, score7, ni);
            max1 = v_f32_max_b(max1, m);
          }
          maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
          float64 origin = v_f32_sel_eq_f32_b(maximum, ni, 0.f, maximum);
          float64 sum0 = 0.f, sum1 = 0.f;
          float64 prob0 = 0.f;
          if (keys > 0) {
            prob0 = tpc_exp_f32_safe(score0 - origin);
            prob0 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 0, prob0, 0.f);
            sum0 += prob0;
          }
          float64 prob1 = 0.f;
          if (keys > 64) {
            prob1 = tpc_exp_f32_safe(score1 - origin);
            prob1 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 64, prob1, 0.f);
            sum1 += prob1;
          }
          float64 prob2 = 0.f;
          if (keys > 128) {
            prob2 = tpc_exp_f32_safe(score2 - origin);
            prob2 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 128, prob2, 0.f);
            sum0 += prob2;
          }
          float64 prob3 = 0.f;
          if (keys > 192) {
            prob3 = tpc_exp_f32_safe(score3 - origin);
            prob3 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 192, prob3, 0.f);
            sum1 += prob3;
          }
          float64 prob4 = 0.f;
          if (keys > 256) {
            prob4 = tpc_exp_f32_safe(score4 - origin);
            prob4 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 256, prob4, 0.f);
            sum0 += prob4;
          }
          float64 prob5 = 0.f;
          if (keys > 320) {
            prob5 = tpc_exp_f32_safe(score5 - origin);
            prob5 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 320, prob5, 0.f);
            sum1 += prob5;
          }
          float64 prob6 = 0.f;
          if (keys > 384) {
            prob6 = tpc_exp_f32_safe(score6 - origin);
            prob6 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 384, prob6, 0.f);
            sum0 += prob6;
          }
          float64 prob7 = 0.f;
          if (keys > 448) {
            prob7 = tpc_exp_f32_safe(score7 - origin);
            prob7 = v_f32_sel_leq_i32_b((int64)lanes, last_key - 448, prob7, 0.f);
            sum1 += prob7;
          }
          sum = v_f32_reduce_add(sum0 + sum1);
          float64 safe = v_f32_sel_grt_f32_b(sum, 0.f, sum, 1.f);
          float64 inverse = v_f32_sel_grt_f32_b(sum, 0.f, v_reciprocal_f32(safe), 0.f);
          if (keys > 0) {
            float128 pair = {prob0 * inverse, prob1 * inverse};
            p[0] = 0;
            v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
          }
          if (keys > 128) {
            float128 pair = {prob2 * inverse, prob3 * inverse};
            p[0] = 128;
            v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
          }
          if (keys > 256) {
            float128 pair = {prob4 * inverse, prob5 * inverse};
            p[0] = 256;
            v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
          }
          if (keys > 384) {
            float128 pair = {prob6 * inverse, prob7 * inverse};
            p[0] = 384;
            v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
          }
        } else {
          for (int k = 0; k < keys; k += 128) { p[0] = k; v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f); }
        }
    }
  }
}
