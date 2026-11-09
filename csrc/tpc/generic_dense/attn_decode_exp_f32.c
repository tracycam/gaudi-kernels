// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"
// Long single-row Decode. Exponentials remain an internal graph tensor.
void main(tensor scores, tensor positions, tensor header, tensor exponent, tensor sums, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 z = {0,0,0,0,0}, h1 = {1,0,0,0,0}, h2 = {2,0,0,0,0};
  int valid_rows = s_i32_ld_g(gen_addr(z, header));
  int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  int key_start = s_i32_ld_g(gen_addr(h2, header));
  int position = s_i32_ld_g(gen_addr(z, positions));
  int visible = valid_rows ? s_i32_max(0, s_i32_min(valid_keys, position-key_start+1)) : 0;
  int keys = get_dim_size(scores, 0);
  uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;
  for (int head = begin[0]; head < end[0]; ++head) {
    float64 max0 = ni, max1 = ni;
    for (int i = 0; i < keys; i += 128) {
      int5 p = {i,0,head,0,0};
      float64 a = v_f32_ld_tnsr_b(p, scores) * scale; p[0] += 64;
      float64 b = v_f32_ld_tnsr_b(p, scores) * scale;
      max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64,visible-i)), a, ni));
      max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64,visible-i-64)), b, ni));
    }
    float64 maximum = v_f32_reduce_max(v_f32_max_b(max0,max1));
    float64 origin = v_f32_sel_eq_f32_b(maximum,ni,0.f,maximum);
    float64 sum0=0.f, sum1=0.f;
    for (int i = 0; i < keys; i += 128) {
      int5 p = {i,0,head,0,0};
      float64 a = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p,scores)*scale-origin); p[0] += 64;
      float64 b = tpc_exp_f32_safe(v_f32_ld_tnsr_b(p,scores)*scale-origin);
      a = v_f32_sel_less_u32_b(lanes,s_i32_max(0,s_i32_min(64,visible-i)),a,0.f);
      b = v_f32_sel_less_u32_b(lanes,s_i32_max(0,s_i32_min(64,visible-i-64)),b,0.f);
      sum0 += a; sum1 += b;
      p[0] -= 64; v_f32_st_tnsr(p,exponent,a); p[0] += 64; v_f32_st_tnsr(p,exponent,b);
    }
    float64 sum = v_f32_reduce_add(sum0+sum1);
    int5 stat = {0,head,0,0,0}; v_f32_st_tnsr(stat,sums,sum);
  }
}
