// Lossless BF16 activation replication + per-task safe-arithmetic guard.
// Normal path: 32*6*max(A) < FP32 max, and BF16 input quantum * .5
// is >= minimum normal FP32. Otherwise the consumer scales each weight first.
void main(tensor input,tensor output,tensor unsafe,int tasks_per_row,int splits) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int kp=get_dim_size(output,0),K=get_dim_size(input,0);
 ushort128 lanes=read_lane_id_2b_b();
 for(int task=begin[0];task<end[0];++task){
  int row=task/tasks_per_row,part=task%splits,start=part*kp;
  ushort128 bad=0;
  for(int k=0;k<kp;k+=128){
   int5 src={start+k,row,0,0,0},dst={k,task,0,0,0};
   bfloat128 x=v_bf16_ld_tnsr_b(src,input);
   ushort128 bits=(ushort128)x;
   int remaining=K-start-k<kp-k?K-start-k:kp-k;
   bits=v_u16_sel_geq_u16_b(lanes,(unsigned short)remaining,0,bits,0,bits,remaining>0);
   if(start+k>=K)bits=0;
   v_bf16_st_tnsr(dst,output,(bfloat128)bits);
   ushort128 mag=bits&32767;
   ushort128 low=v_u16_sel_less_u16_b(mag,1152,1,0);
   low=v_u16_sel_eq_u16_b(mag,0,0,low);
   bad|=low|v_u16_sel_grt_u16_b(mag,31488,1,0);
  }
  bad=(ushort128)v_i16_reduce_max((short128)bad);
  int5 f={0,task,0,0,0};v_u16_st_tnsr_partial(f,unsafe,bad,0,0);
 }
}
