// SPDX-License-Identifier: Apache-2.0
#include "../moe_router/precise_ratio.h"
void main(tensor gate_up,tensor activation){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int chunk=begin[0];chunk<end[0];++chunk){
  int5 p={chunk*128,0,0,0,0},up={chunk*128+2048,0,0,0,0};
  float64 g0=v_f32_ld_tnsr_b(p,gate_up),u0=v_f32_ld_tnsr_b(up,gate_up);
  p[0]+=64;up[0]+=64;
  float64 g1=v_f32_ld_tnsr_b(p,gate_up),u1=v_f32_ld_tnsr_b(up,gate_up);
  float64 a0=router_precise_ratio(g0,1.0f+v_exp_f32(-g0))*u0;
  float64 a1=router_precise_ratio(g1,1.0f+v_exp_f32(-g1))*u1;
  int5 dst={chunk*128,0,0,0,0};v_f32_st_tnsr(dst,activation,a0);
  dst[0]+=64;v_f32_st_tnsr(dst,activation,a1);
 }
}
