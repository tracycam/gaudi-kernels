// Offline candidate: two independent tokens; original KV owner/layout.
#define WINDOW_METADATA 1
#define AV_SHUFFLE 1
// FP32 score/softmax/AV candidate. Only BF16 inputs and final BF16 output.
// Same M1/two-physical-page address contract as head.c; no KV intermediates.
#ifndef WINDOW_METADATA
#define WINDOW_METADATA 0
#endif
#ifndef AV_SHUFFLE
#define AV_SHUFFLE 0
#endif
#if AV_SHUFFLE
#define ALL_GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
static inline float64 broadcast_probability(float64 source,int index){
 // Byte SHUFFLE needs bit7 enabled. Each 64-byte dual group contributes its
 // selected FP32 word, then one exact dual-group move broadcasts the right group.
 uint64 control=0x83828180u+(unsigned)(index&15)*0x04040404u;
 float64 selected=(float64)v_u8_shuffle_b((uchar256)source,(uchar256)control,0,0);
 int group=(index>>4)&3;
 if(group==0)return v_f32_mov_dual_group_all_b(selected,-1,0,0,0,0,ALL_GROUPS,0);
 if(group==1)return v_f32_mov_dual_group_all_b(selected,-1,1,1,1,1,ALL_GROUPS,0);
 if(group==2)return v_f32_mov_dual_group_all_b(selected,-1,2,2,2,2,ALL_GROUPS,0);
 return v_f32_mov_dual_group_all_b(selected,-1,3,3,3,3,ALL_GROUPS,0);
}
#endif
static inline float64 dot192(float64 q0,float64 q1,float64 q2,tensor key,int slot){
 int5 p={0,0,slot,0,0};bfloat128 k=v_bf16_ld_tnsr_b(p,key);p[0]=128;bfloat128 tail=v_bf16_ld_tnsr_partial_b(p,key,63,0);
 float128 kf=convert_bfloat128_to_float128(k,SW_LINEAR),tf=convert_bfloat128_to_float128(tail,SW_LINEAR);
 float64 acc=q0*kf.v1;acc=v_f32_mac_b(q1,kf.v2,acc);acc=v_f32_mac_b(q2,tf.v1,acc);return v_f32_reduce_add(acc);
}
static inline float128 dot192_pair(float64 q0,float64 q1,float64 q2,tensor key,int slot){
 int5 a={0,0,slot,0,0},b={0,0,slot+1,0,0};
 bfloat128 ka=v_bf16_ld_tnsr_b(a,key),kb=v_bf16_ld_tnsr_b(b,key);
 a[0]=128;b[0]=128;
 bfloat128 ta=v_bf16_ld_tnsr_partial_b(a,key,63,0),tb=v_bf16_ld_tnsr_partial_b(b,key,63,0);
 float128 af=convert_bfloat128_to_float128(ka,SW_LINEAR),bf=convert_bfloat128_to_float128(kb,SW_LINEAR);
 float64 at=convert_bfloat128_to_float128(ta,SW_LINEAR).v1,bt=convert_bfloat128_to_float128(tb,SW_LINEAR).v1;
 float64 aa=q0*af.v1,ab=q0*bf.v1;
 aa=v_f32_mac_b(q1,af.v2,aa);ab=v_f32_mac_b(q1,bf.v2,ab);
 aa=v_f32_mac_b(q2,at,aa);ab=v_f32_mac_b(q2,bt,ab);
 float128 out={v_f32_reduce_add(aa),v_f32_reduce_add(ab)};return out;
}
void main(tensor query,tensor key,tensor value,tensor page_ids,
#if WINDOW_METADATA
 tensor groups,
#else
 tensor starts,
#endif
 tensor position,tensor sinks,tensor output,float query_scale){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size(),zero={0,0,0,0,0};
 int pos=s_i32_ld_g(gen_addr(zero,position)),slots=get_dim_size(key,2);float ni=-1.0f/0.0f;
 int5 one={1,0,0,0,0};
#if WINDOW_METADATA
 int logical0=(s_i32_max(0,pos-127)>>7)<<7,logical1=logical0+128;
 int physical0=s_i32_ld_g(gen_addr(zero,page_ids)),physical1=-1;
 int group0=s_i32_ld_g(gen_addr(zero,groups));if(group0!=0)physical0=-1;
 if(pos>=logical1 && get_dim_size(page_ids,0)>1){physical1=s_i32_ld_g(gen_addr(one,page_ids));int group1=s_i32_ld_g(gen_addr(one,groups));if(group1!=0)physical1=-1;}
#else
 int physical0=s_i32_ld_g(gen_addr(zero,page_ids)),physical1=s_i32_ld_g(gen_addr(one,page_ids));
 int logical0=s_i32_ld_g(gen_addr(zero,starts)),logical1=s_i32_ld_g(gen_addr(one,starts));
#endif
 for(int head=begin[0];head<end[0];++head){
  bool flat=get_dim_size(query,0)!=192;int5 qp={flat?head*192:0,flat?0:head,0,0,0};
  float128 q=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(qp,query),SW_LINEAR);qp[0]+=128;
  float64 tail=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(qp,query,63,0),SW_LINEAR).v1;
  float64 score0=ni,score1=ni;uint64 lane=read_lane_id_4b_b();
  for(int page=0;page<2;++page){
   int physical=page?physical1:physical0,logical=page?logical1:logical0;
   bool valid=physical>=0&&physical<slots/128&&logical>=0&&logical<=2147483520&&pos>=0;
   int lo=0,hi=0;if(valid){lo=s_i32_min(128,s_i32_max(0,pos-127-logical));hi=s_i32_max(0,s_i32_min(128,pos-logical+1));}
   int t=lo;
   for(;t+1<hi;t+=2){
    int index=logical+t-(pos-127);
    float128 scores=dot192_pair(q.v1,q.v2,tail,key,physical*128+t);
    float64 a=scores.v1*query_scale,b=scores.v2*query_scale;
    if(index<64)score0=v_f32_sel_eq_u32_b(lane,index,a,score0);else score1=v_f32_sel_eq_u32_b(lane,index-64,a,score1);
    ++index;
    if(index<64)score0=v_f32_sel_eq_u32_b(lane,index,b,score0);else score1=v_f32_sel_eq_u32_b(lane,index-64,b,score1);
   }
   if(t<hi){
    int index=logical+t-(pos-127);float64 score=dot192(q.v1,q.v2,tail,key,physical*128+t)*query_scale;
    if(index<64)score0=v_f32_sel_eq_u32_b(lane,index,score,score0);else score1=v_f32_sel_eq_u32_b(lane,index-64,score,score1);
   }
  }
  int5 hp={head,0,0,0,0};float64 sink=(float)s_bf16_ld_g(gen_addr(hp,sinks));
  float64 maximum=v_f32_max_b(v_f32_reduce_max(v_f32_max_b(score0,score1)),sink),origin=v_f32_sel_eq_f32_b(maximum,ni,0,maximum);
  float64 p0=v_exp_f32(score0-origin),p1=v_exp_f32(score1-origin);p0=v_f32_sel_eq_f32_b(score0,ni,0,p0);p1=v_f32_sel_eq_f32_b(score1,ni,0,p1);
  float64 sink_p=v_exp_f32(sink-origin);sink_p=v_f32_sel_eq_f32_b(sink,ni,0,sink_p);
  float64 denominator=v_f32_reduce_add(p0+p1)+sink_p;denominator=v_f32_sel_leq_f32_b(denominator,0,1,denominator);
  float128 sum={0,0};
  for(int page=0;page<2;++page){
   int physical=page?physical1:physical0,logical=page?logical1:logical0;
   bool valid=physical>=0&&physical<slots/128&&logical>=0&&logical<=2147483520&&pos>=0;
   int lo=0,hi=0;if(valid){lo=s_i32_min(128,s_i32_max(0,pos-127-logical));hi=s_i32_max(0,s_i32_min(128,pos-logical+1));}
   for(int t=lo;t<hi;++t){
    int index=logical+t-(pos-127);float64 probabilities=index<64?p0:p1;
    int5 vp={0,0,physical*128+t,0,0};bfloat128 loaded=v_bf16_ld_tnsr_b(vp,value);
#if AV_SHUFFLE
    float64 p=broadcast_probability(probabilities,index);
#else
    float64 p=v_f32_reduce_add(v_f32_sel_eq_u32_b(lane,index&63,probabilities,0));
#endif
    float128 v=v_convert_bf16_to_f32_all_b(loaded);
    sum.v1=v_f32_mac_b(v.v1,p,sum.v1);sum.v2=v_f32_mac_b(v.v2,p,sum.v2);
   }
  }
  float64 reciprocal=v_reciprocal_f32(denominator);sum.v1*=reciprocal;sum.v2*=reciprocal;
  int5 dst={0,head,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(sum,SW_RHNE));
 }
}
