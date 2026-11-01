// SPDX-License-Identifier: Apache-2.0
// Keep gate and up projections in FP32 through SiLU and multiplication;
// the declared expert activation is rounded to BF16 once on output.
#include "../common/tpc_math_safe.h"
void main(tensor partial, tensor ids, tensor output, int splits) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int width=get_dim_size(output,0);
 for(int slot=begin[0];slot<end[0];++slot){
  for(int row=0;row<width;row+=128){
   int gate_row=slot*2*width+row,up_row=gate_row+width;
   int gate_base=(gate_row/512)*splits*512+gate_row%512;
   int up_base=(up_row/512)*splits*512+up_row%512;
   float128 gate={0},up={0};
   for(int split=0;split<splits;++split){
    int5 gp={gate_base+split*512,0,0,0,0},u={up_base+split*512,0,0,0,0};
    gate.v1+=v_f32_ld_tnsr_b(gp,partial);gp[0]+=64;gate.v2+=v_f32_ld_tnsr_b(gp,partial);
    up.v1+=v_f32_ld_tnsr_b(u,partial);u[0]+=64;up.v2+=v_f32_ld_tnsr_b(u,partial);
   }
   gate.v1=gate.v1*v_reciprocal_f32(1.f+tpc_exp_f32_safe(-gate.v1));
   gate.v2=gate.v2*v_reciprocal_f32(1.f+tpc_exp_f32_safe(-gate.v2));
   gate.v1*=up.v1;
   gate.v2*=up.v2;
   int5 dst={row,slot,0,0,0};
   v_bf16_st_tnsr(dst,output,v_convert_f32_to_bf16_all_b(gate));
  }
 }
}
