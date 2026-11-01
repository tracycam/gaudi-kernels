// SPDX-License-Identifier: Apache-2.0
// Input [ranks * M, H] BF16, output [M, H] BF16. Round only after FP32 sum.
// SW_LINEAR is necessary: hardware convert-all interleaves the two FP32 registers.
void main(tensor input, tensor output, int ranks) {
 int5 b=get_index_space_offset(), end=b+get_index_space_size();
 int h=get_dim_size(output,0), m=get_dim_size(output,1);
 for(int task=b[0];task<end[0];++task){
  int row=task/(h/128), col=(task%(h/128))*128;
  float128 acc={0};
  for(int rank=0;rank<ranks;++rank){
   int5 p={col,rank*m+row,0,0,0};
   float128 v=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   acc.v1+=v.v1;acc.v2+=v.v2;
  }
  bfloat128 result=convert_float128_to_bfloat128(acc,SW_LINEAR|SW_RHNE);
  int5 dst={col,row,0,0,0};v_bf16_st_tnsr(dst,output,result);
 }
}
