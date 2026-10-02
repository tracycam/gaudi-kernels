// Preserves precision_ops sum_ff: original FP32 inputs and ordered FP32 adds.
void main(tensor input, tensor output, int ranks) {
 int5 b=get_index_space_offset(), end=b+get_index_space_size();
 int h=get_dim_size(output,0), m=get_dim_size(output,1);
 for(int task=b[0];task<end[0];++task){
  int row=task/(h/128), col=(task%(h/128))*128;
  float128 acc={0};
  for(int rank=0;rank<ranks;++rank){
   int5 p={col,rank*m+row,0,0,0};
   acc.v1+=v_f32_ld_tnsr_b(p,input);p[0]+=64;
   acc.v2+=v_f32_ld_tnsr_b(p,input);
  }
  int5 dst={col,row,0,0,0};v_f32_st_tnsr(dst,output,acc.v1);dst[0]+=64;v_f32_st_tnsr(dst,output,acc.v2);
 }
}
