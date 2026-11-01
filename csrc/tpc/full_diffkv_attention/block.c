// SPDX-License-Identifier: Apache-2.0
static inline float64 dot192(float64 q0,float64 q1,float64 q2,tensor key,int slot){
 int5 p={0,0,slot,0,0};bfloat128 k=v_bf16_ld_tnsr_b(p,key);p[0]=128;bfloat128 tail=v_bf16_ld_tnsr_partial_b(p,key,63,0);
 float128 kf=convert_bfloat128_to_float128(k,SW_LINEAR),tf=convert_bfloat128_to_float128(tail,SW_LINEAR);
 float64 acc=q0*kf.v1;acc=v_f32_mac_b(q1,kf.v2,acc);acc=v_f32_mac_b(q2,tf.v1,acc);
 return v_f32_reduce_add(acc);
}
void main(tensor query,tensor key,tensor value,tensor page_ids,tensor block_usage,
          tensor block_groups,tensor block_max,tensor block_sums,tensor block_values,
          float query_scale){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int total_slots=get_dim_size(key,2);float ni=-1.0f/0.0f;
 for(int page=begin[0];page<end[0];++page){
  for(int head=begin[1];head<end[1];++head){
   int5 qp={0,head,0,0,0};bfloat128 qraw=v_bf16_ld_tnsr_b(qp,query);
   float128 q=convert_bfloat128_to_float128(qraw,SW_LINEAR);
   qp[0]=128;bfloat128 tailraw=v_bf16_ld_tnsr_partial_b(qp,query,63,0);
   float128 tail=convert_bfloat128_to_float128(tailraw,SW_LINEAR);
   float64 q0=q.v1*query_scale,q1=q.v2*query_scale,q2=tail.v1*query_scale;
   int5 ip={page,0,0,0,0};int physical=s_i32_ld_g(gen_addr(ip,page_ids));
   int usage=s_i32_ld_g(gen_addr(ip,block_usage)),group=s_i32_ld_g(gen_addr(ip,block_groups));
   bool valid=group==0&&physical>=0&&physical<total_slots/128&&usage>0;
   int hi=valid?s_i32_min(usage,128):0;
   float64 score0=ni,score1=ni;uint64 lane=read_lane_id_4b_b();
   for(int t=0;t<hi;++t){
    float64 score=dot192(q0,q1,q2,key,physical*128+t);
    if(t<64)score0=v_f32_sel_eq_u32_b(lane,t,score,score0);else score1=v_f32_sel_eq_u32_b(lane,t-64,score,score1);
   }
   float64 maximum=v_f32_reduce_max(v_f32_max_b(score0,score1));
   float64 origin=v_f32_sel_eq_f32_b(maximum,ni,0,maximum);
   float64 p0=v_exp_f32(score0-origin),p1=v_exp_f32(score1-origin);
   p0=v_f32_sel_eq_f32_b(score0,ni,0,p0);p1=v_f32_sel_eq_f32_b(score1,ni,0,p1);
   float64 sum=v_f32_reduce_add(p0+p1);
   float128 accum={0,0};
   for(int t=0;t<hi;++t){
    float64 probabilities=t<64?p0:p1;
    float64 probability=v_f32_reduce_add(v_f32_sel_eq_u32_b(lane,t&63,probabilities,0));
    int5 vp={0,0,physical*128+t,0,0};float128 v=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vp,value));
    accum.v1=v_f32_mac_b(v.v1,probability,accum.v1);accum.v2=v_f32_mac_b(v.v2,probability,accum.v2);
   }
   int5 md={page*16+head,0,0,0,0};
   v_f32_st_tnsr_partial(md,block_max,maximum,0,0);
   v_f32_st_tnsr_partial(md,block_sums,sum,0,0);
   int5 dst={0,page,head,0,0};v_f32_st_tnsr(dst,block_values,accum.v1);
   dst[0]=64;v_f32_st_tnsr(dst,block_values,accum.v2);
  }
 }
}
