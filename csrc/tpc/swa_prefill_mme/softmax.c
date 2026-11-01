// SPDX-License-Identifier: Apache-2.0
// FP32 causal 128-window softmax including the per-head sink. Split the
// probabilities into BF16 high/low parts for two MME PV products.
#include "../common/tpc_math_safe.h"
void main(tensor scores,tensor positions,tensor sinks,tensor high,tensor low,float scale){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 uint64 lane=read_lane_id_4b_b();float ni=-1.0f/0.0f;
 for(int batch=begin[1];batch<end[1];++batch){
  int5 p0={batch*128,0,0,0,0};int first=s_i32_max(0,s_i32_ld_g(gen_addr(p0,positions))-127)>>7;
  for(int col=begin[0];col<end[0];++col){
   int row=batch*128+col/16;int5 rp={row,0,0,0,0};int pos=s_i32_ld_g(gen_addr(rp,positions));
   int5 hp={col%16,0,0,0,0};float64 sink=(float)s_bf16_ld_g(gen_addr(hp,sinks));
   float64 mx=sink;
   for(int block=0;block<6;++block){int start=first*128+block*64;
    unsigned lo=s_i32_max(0,s_i32_min(64,pos-127-start));
    unsigned hi=s_i32_max(0,s_i32_min(64,pos+1-start));
    int5 q={block*64,col,batch,0,0};float64 v=v_f32_ld_tnsr_b(q,scores)*scale;
    v=v_f32_sel_less_u32_b(lane,lo,ni,v);v=v_f32_sel_less_u32_b(lane,hi,v,ni);
    mx=v_f32_max_b(mx,v);
   }
   float64 maximum=v_f32_reduce_max(mx),sum=0;
   for(int block=0;block<6;++block){int start=first*128+block*64;
    unsigned lo=s_i32_max(0,s_i32_min(64,pos-127-start));
    unsigned hi=s_i32_max(0,s_i32_min(64,pos+1-start));
    int5 q={block*64,col,batch,0,0};float64 v=tpc_exp_f32_safe(v_f32_ld_tnsr_b(q,scores)*scale-maximum);
    v=v_f32_sel_less_u32_b(lane,lo,0.f,v);v=v_f32_sel_less_u32_b(lane,hi,v,0.f);sum+=v;
   }
   // sink was broadcast in every lane, so add it after reducing score lanes.
   float64 sink_exp=tpc_exp_f32_safe(sink-maximum);
   float64 denom=v_f32_reduce_add(sum)+sink_exp;
   float64 reciprocal=v_reciprocal_f32(denom);
   for(int block=0;block<3;++block){float128 pair;
    for(int part=0;part<2;++part){int index=block*2+part,start=first*128+index*64;
     unsigned lo=s_i32_max(0,s_i32_min(64,pos-127-start));
     unsigned hi=s_i32_max(0,s_i32_min(64,pos+1-start));
     int5 q={index*64,col,batch,0,0};float64 v=tpc_exp_f32_safe(v_f32_ld_tnsr_b(q,scores)*scale-maximum)*reciprocal;
     v=v_f32_sel_less_u32_b(lane,lo,0.f,v);v=v_f32_sel_less_u32_b(lane,hi,v,0.f);
     if(part==0)pair.v1=v;else pair.v2=v;
    }
    bfloat128 upper=convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE);
    float128 rounded=convert_bfloat128_to_float128(upper,SW_LINEAR);
    float128 residual={pair.v1-rounded.v1,pair.v2-rounded.v2};
    int5 dst={block*128,col,batch,0,0};v_bf16_st_tnsr(dst,high,upper);
    v_bf16_st_tnsr(dst,low,convert_float128_to_bfloat128(residual,SW_LINEAR|SW_RHNE));
   }
  }
 }
}
