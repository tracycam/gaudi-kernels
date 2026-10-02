// SPDX-License-Identifier: Apache-2.0
// Whole-row ownership follows 1cat control_gemv (9f2b52a). BF16 operands remain
// BF16 in HBM and registers; native BF16 MAC accumulates in FP32. No split-K
// tensor, atomic, standalone bias node, or intermediate BF16 rounding.
void main(tensor activation,tensor weight,
#if BIAS
 tensor bias,
#endif
 tensor output) {
 int5 start=get_index_space_offset(),end=start+get_index_space_size();
 int width=get_dim_size(activation,0);
 for(int token=start[1];token<end[1];++token)for(int row=start[0];row<end[0];++row){
#if UNROLL4
  float128 acc0={0,0},acc1={0,0},acc2={0,0},acc3={0,0};
  for(int k=0;k<width;k+=512){
#define STEP(R) {int5 a={k+128*R,token,0,0,0},w={k+128*R,row,0,0,0};bool live=k+128*R<width;acc##R=v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(a,activation,0,0,live),v_bf16_ld_tnsr_b(w,weight,0,0,live),acc##R);}
   STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
  }
  float128 acc={acc0.v1+acc1.v1+acc2.v1+acc3.v1,acc0.v2+acc1.v2+acc2.v2+acc3.v2};
#else
  float128 acc={0,0};
  for(int k=0;k<width;k+=128){
   int5 a={k,token,0,0,0},w={k,row,0,0,0};
   acc=v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(a,activation),v_bf16_ld_tnsr_b(w,weight),acc);
  }
#endif
  float64 result=v_f32_reduce_add(acc.v1+acc.v2);
#if BIAS
  int5 b={row,0,0,0,0};result+=v_f32_ld_g(gen_addr(b,bias));
#endif
  int5 y={row,token,0,0,0};
#if OUTPUT_BF16
  float128 pair={result,result};v_bf16_st_tnsr_partial(y,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE),0,0);
#else
  v_f32_st_tnsr_partial(y,output,result,0,0);
#endif
 }
}
