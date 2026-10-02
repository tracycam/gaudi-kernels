// Experimental post-TopK fusion. Vendor TopK owns selection AND output order.
// Original FP32 scores are gathered; correction bias is never used as weight.
// Scalar loads make arbitrary expert IDs explicit; no BF16 intermediate.
#include "precise_ratio.h"
void main(tensor scores,tensor ids,tensor weights,tensor ids_out,tensor sum_out,
          float factor,int renormalize,int apply_scale) {
 uint64 lane=read_lane_id_4b_b();float64 values=0;int64 indices=0;
 #pragma loop_unroll(8)
 for(int slot=0;slot<8;++slot) {
  int5 p={slot,0,0,0,0};int expert=s_i32_ld_g(gen_addr(p,ids));
  int5 q={expert,0,0,0,0};float value=s_f32_ld_g(gen_addr(q,scores));
  values=v_f32_sel_eq_u32_b(lane,slot,value,values);
  indices=v_i32_sel_eq_u32_b(lane,slot,expert,indices);
 }
 float64 total=v_f32_reduce_add(values);int5 out={0,0,0,0,0};
 v_f32_st_tnsr_partial(out,sum_out,total,0,0);
 if(renormalize) {
  values=router_precise_ratio(values,total);
 }
 // Explicit predicate preserves the factor==1 no-arithmetic branch.
 values=v_f32_mul_b(values,factor,0,values,apply_scale!=0);
 v_f32_st_tnsr_partial(out,weights,values,7,0);
 v_i32_st_tnsr_partial(out,ids_out,indices,7,0);
}
