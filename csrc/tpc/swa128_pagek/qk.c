// Offline score-only prototype: lane=token, same BF16 K capacity, FP32 MAC.
// K tensor native [128,192,pages]; meta [physical_page,lo,count].
#define GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
static inline bfloat128 broadcast_q(bfloat128 q,int d){
 ushort128 control=(unsigned short)(0x8180u+(unsigned)(d&31)*0x0202u);
 bfloat128 v=(bfloat128)v_u8_shuffle_b((uchar256)q,(uchar256)control,0,0);
 int group=(d>>5)&3;
 if(group==0)return v_bf16_mov_dual_group_all_b(v,-1,0,0,0,0,GROUPS,0);
 if(group==1)return v_bf16_mov_dual_group_all_b(v,-1,1,1,1,1,GROUPS,0);
 if(group==2)return v_bf16_mov_dual_group_all_b(v,-1,2,2,2,2,GROUPS,0);
 return v_bf16_mov_dual_group_all_b(v,-1,3,3,3,3,GROUPS,0);
}
void main(tensor query,tensor key,tensor metadata,tensor output){
 int5 start=get_index_space_offset(),end=start+get_index_space_size(),mp={0,0,0,0,0};
 int page=s_i32_ld_g(gen_addr(mp,metadata));mp[0]=1;int lo=s_i32_ld_g(gen_addr(mp,metadata));mp[0]=2;int count=s_i32_ld_g(gen_addr(mp,metadata));
 bool valid=page>=0&&page<get_dim_size(key,2)&&lo>=0&&count>0&&count<=128&&lo+count<=128;
 for(int h=start[0];h<end[0];++h){
  int5 qp={0,h,0,0,0};bfloat128 q0=v_bf16_ld_tnsr_b(qp,query);qp[0]=128;
  bfloat128 q1=v_bf16_ld_tnsr_partial_b(qp,query,63,0);float128 acc={0,0};
  if(valid)for(int d=0;d<192;++d){
   bfloat128 q=broadcast_q(d<128?q0:q1,d&127);int5 kp={lo,d,page,0,0};
   bfloat128 k=v_bf16_ld_tnsr_partial_b(kp,key,count-1,0);
   acc=v_bf16_mac_acc32_b(k,q,acc);
  }
  // Explicit native score layout: first64=even tokens, second64=odd tokens.
  // A future AV reader uses (t&1 ? v2 : v1) with lane t>>1; no BF16 roundtrip.
  int5 dst={0,h,0,0,0};v_f32_st_tnsr(dst,output,acc.v1);dst[0]=64;v_f32_st_tnsr(dst,output,acc.v2);
 }
}
