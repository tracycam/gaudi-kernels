// Offline prototype: one decode query/head/task, two selected physical pages.
// Native Q[192,H,1], K[192,1,T], V[128,1,T], I32 page_ids[2],
// logical_starts[2], position[1], BF16 sinks[H], Y[128,H,1]. H=16 production.
// Scores/probabilities occupy two vector registers per page; no HBM scratch.
#ifndef SWA_DEBUG_STAGE
#define SWA_DEBUG_STAGE 0
#endif
static inline float64 rounded(float64 x){
 float128 both={x,x};bfloat128 b=convert_float128_to_bfloat128(both,SW_RHNE);return v_convert_bf16_to_f32_all_b(b).v1;
}
static inline float64 dot192(float64 q0,float64 q1,float64 q2,tensor key,int slot){
 int5 p={0,0,slot,0,0};bfloat128 k=v_bf16_ld_tnsr_b(p,key);p[0]=128;bfloat128 tail=v_bf16_ld_tnsr_partial_b(p,key,63,0);
 float128 kf=convert_bfloat128_to_float128(k,SW_LINEAR),tf=convert_bfloat128_to_float128(tail,SW_LINEAR);
 float64 acc=q0*kf.v1;acc=v_f32_mac_b(q1,kf.v2,acc);acc=v_f32_mac_b(q2,tf.v1,acc);
 return v_f32_reduce_add(acc);
}
void main(tensor query,tensor key,tensor value,tensor page_ids,tensor starts,
          tensor position,tensor sinks,tensor output,float query_scale){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int5 z={0,0,0,0,0};
 int pos=s_i32_ld_g(gen_addr(z,position)),total_slots=get_dim_size(key,2);float negative_infinity=-1.0f/0.0f;
 for(int head=begin[0];head<end[0];++head){
  bool flat_query=get_dim_size(query,0)!=192;
  int5 qp={flat_query?head*192:0,flat_query?0:head,0,0,0};bfloat128 q=v_bf16_ld_tnsr_b(qp,query);qp[0]+=128;bfloat128 qt=v_bf16_ld_tnsr_partial_b(qp,query,63,0);
  float128 qf=v_convert_bf16_to_f32_all_b(q);qf.v1*=query_scale;qf.v2*=query_scale;q=convert_float128_to_bfloat128(qf,SW_RHNE);
  qf=v_convert_bf16_to_f32_all_b(qt);qf.v1*=query_scale;qf.v2*=query_scale;qt=convert_float128_to_bfloat128(qf,SW_RHNE);
  float128 linear_q=convert_bfloat128_to_float128(q,SW_LINEAR),linear_tail=convert_bfloat128_to_float128(qt,SW_LINEAR);
  float64 q0=linear_q.v1,q1=linear_q.v2,q2=linear_tail.v1;
  int5 hp={head,0,0,0,0};float64 sink=(float)s_bf16_ld_g(gen_addr(hp,sinks));
  float64 max0=negative_infinity,max1=negative_infinity,sum_page0=0,sum_page1=0;
  bfloat128 context0=0,context1=0;
  for(int page=0;page<2;++page){
   int5 ip={page,0,0,0,0};int physical=s_i32_ld_g(gen_addr(ip,page_ids)),logical=s_i32_ld_g(gen_addr(ip,starts));
   bool valid=physical>=0 && physical<total_slots/128 && logical>=0 && logical<=2147483520 && pos>=0;
   int lo=0,hi=0;
   if(valid){lo=s_i32_min(128,s_i32_max(0,pos-127-logical));hi=s_i32_max(0,s_i32_min(128,pos-logical+1));}
#if SWA_DEBUG_STAGE == 0
   if(lo>=hi){if(page==0){max0=sink;sum_page0=0;context0=0;}else{max1=sink;sum_page1=0;context1=0;}continue;}
#endif
   float64 score0=negative_infinity,score1=negative_infinity;uint64 lane=read_lane_id_4b_b();
   for(int t=lo;t<hi;++t){
    float64 score=rounded(dot192(q0,q1,q2,key,physical*128+t));
    if(t<64)score0=v_f32_sel_eq_u32_b(lane,t,score,score0);else score1=v_f32_sel_eq_u32_b(lane,t-64,score,score1);
   }
   #if SWA_DEBUG_STAGE == 1
   if(page==0){float128 debug={score0,score1};int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(debug,SW_RHNE|SW_LINEAR));continue;}
#endif
   float64 maximum=v_f32_max_b(v_f32_reduce_max(v_f32_max_b(score0,score1)),sink);
   if(page==0)max0=maximum;else max1=maximum;float64 origin=v_f32_sel_eq_f32_b(maximum,negative_infinity,0,maximum);
   float64 probability0=rounded(v_exp_f32(rounded(score0-origin))),probability1=rounded(v_exp_f32(rounded(score1-origin)));
   probability0=v_f32_sel_eq_f32_b(score0,negative_infinity,0,probability0);probability1=v_f32_sel_eq_f32_b(score1,negative_infinity,0,probability1);
   #if SWA_DEBUG_STAGE == 2
   if(page==0){float128 debug={probability0,probability1};int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(debug,SW_RHNE|SW_LINEAR));continue;}
#endif
   float64 sum=v_f32_reduce_add(probability0+probability1);float128 accum={0,0};
   for(int t=lo;t<hi;++t){
    {
     float64 probabilities=t<64?probability0:probability1;
     float64 probability=v_f32_reduce_add(v_f32_sel_eq_u32_b(lane,t&63,probabilities,0));
     int5 vp={0,0,physical*128+t,0,0};float128 v=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vp,value));
     accum.v1=v_f32_mac_b(v.v1,probability,accum.v1);accum.v2=v_f32_mac_b(v.v2,probability,accum.v2);
    }
   }
   bfloat128 page_result=convert_float128_to_bfloat128(accum,SW_RHNE);
   if(page==0){sum_page0=rounded(sum);context0=page_result;}else{sum_page1=rounded(sum);context1=page_result;}
#if SWA_DEBUG_STAGE == 3
   if(page==0){int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,page_result);}
#endif
  }
  float64 maximum=v_f32_max_b(max0,max1),origin=v_f32_sel_eq_f32_b(maximum,negative_infinity,0,maximum);
  float64 a0=rounded(v_exp_f32(rounded(max0-origin))),a1=rounded(v_exp_f32(rounded(max1-origin)));
  a0=v_f32_sel_eq_f32_b(max0,negative_infinity,0,a0);a1=v_f32_sel_eq_f32_b(max1,negative_infinity,0,a1);
  float64 sum0=rounded(sum_page0*a0),sum1=rounded(sum_page1*a1),denominator=rounded(sum0+sum1);
  float64 sink_weight=rounded(v_exp_f32(rounded(sink-origin)));sink_weight=v_f32_sel_eq_f32_b(sink,negative_infinity,0,sink_weight);
  denominator=rounded(denominator+sink_weight);
  float64 d0=v_f32_max_b(denominator,sum0),d1=v_f32_max_b(denominator,sum1);d0=v_f32_sel_leq_f32_b(d0,0,1,d0);d1=v_f32_sel_leq_f32_b(d1,0,1,d1);
  // Candidate reciprocal multiply: native vendor div numerical gate outstanding.
  float64 r0=rounded(a0*v_reciprocal_f32(d0)),r1=rounded(a1*v_reciprocal_f32(d1));
  #if SWA_DEBUG_STAGE == 4
  uint64 di=read_lane_id_4b_b();float64 d=0;
  d=v_f32_sel_eq_u32_b(di,0,max0,d);d=v_f32_sel_eq_u32_b(di,1,max1,d);
  d=v_f32_sel_eq_u32_b(di,2,sum_page0,d);d=v_f32_sel_eq_u32_b(di,3,sum_page1,d);
  d=v_f32_sel_eq_u32_b(di,4,a0,d);d=v_f32_sel_eq_u32_b(di,5,a1,d);
  d=v_f32_sel_eq_u32_b(di,6,denominator,d);d=v_f32_sel_eq_u32_b(di,7,r0,d);d=v_f32_sel_eq_u32_b(di,8,r1,d);
  float128 debug={d,d};int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(debug,SW_LINEAR|SW_RHNE));
#endif
  float128 c0=v_convert_bf16_to_f32_all_b(context0),c1=v_convert_bf16_to_f32_all_b(context1);
  c0.v1*=r0;c0.v2*=r0;c1.v1*=r1;c1.v2*=r1;
  bfloat128 b0=convert_float128_to_bfloat128(c0,SW_RHNE),b1=convert_float128_to_bfloat128(c1,SW_RHNE);
  c0=v_convert_bf16_to_f32_all_b(b0);c1=v_convert_bf16_to_f32_all_b(b1);c0.v1+=c1.v1;c0.v2+=c1.v2;
  #if SWA_DEBUG_STAGE == 0
  int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(c0,SW_RHNE));
#endif
 }
}
