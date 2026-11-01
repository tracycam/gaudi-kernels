// SPDX-License-Identifier: Apache-2.0
#include "../moe_router/precise_ratio.h"
void main(tensor block_max,tensor block_sums,tensor block_values,tensor output){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int pages=get_dim_size(block_max,0)/16;float ni=-1.0f/0.0f;
 for(int head=begin[0];head<end[0];++head){
  float64 maximum=ni;
  for(int page=0;page<pages;++page){
   int5 p={page*16+head,0,0,0,0};maximum=v_f32_max_b(maximum,s_f32_ld_g(gen_addr(p,block_max)));
  }
  float64 origin=v_f32_sel_eq_f32_b(maximum,ni,0,maximum),denominator=0;
  for(int page=0;page<pages;++page){
   int5 md={page*16+head,0,0,0,0};float64 local_max=s_f32_ld_g(gen_addr(md,block_max));
   float64 adjustment=v_exp_f32(local_max-origin);
   adjustment=v_f32_sel_eq_f32_b(local_max,ni,0,adjustment);
   float64 local_sum=s_f32_ld_g(gen_addr(md,block_sums));
   denominator+=local_sum*adjustment;
  }
  float128 context={0,0};
  for(int page=0;page<pages;++page){
   int5 md={page*16+head,0,0,0,0};float64 local_max=s_f32_ld_g(gen_addr(md,block_max));
   float64 adjustment=v_exp_f32(local_max-origin);
   adjustment=v_f32_sel_eq_f32_b(local_max,ni,0,adjustment);
   float64 local_sum=s_f32_ld_g(gen_addr(md,block_sums));
   float64 adjusted=local_sum*adjustment;
   float64 safe=v_f32_max_b(denominator,adjusted);safe=v_f32_sel_leq_f32_b(safe,0,1,safe);
   float64 rescale=router_precise_ratio(adjustment,safe);
   int5 src={0,page,head,0,0};float64 value0=v_f32_ld_tnsr_b(src,block_values);
   src[0]=64;float64 value1=v_f32_ld_tnsr_b(src,block_values);
   context.v1=v_f32_mac_b(value0,rescale,context.v1);
   context.v2=v_f32_mac_b(value1,rescale,context.v2);
  }
  int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(context,SW_RHNE));
 }
}
