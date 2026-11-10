// FP8 lane packing follows 1cat-vllm-gaudi 9f2b52a dense_quant_gaudi2.c
// (Apache-2.0); our block128 quantizer, scale policy and subnormal policy differ.
// Original E4M3FN activation contract. One task per row and 128-value block.
// Store exact FP8 values in BF16, or their FP32 dequantization for full-K MME.
#ifndef DEQUANTIZE
#define DEQUANTIZE 0
#endif
#include "activation_quant_common.h"
void main(tensor input,tensor output,tensor scales) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[1];row<end[1];++row) {
  for(int g=begin[0];g<end[0];++g) {
   int5 p={g*128,row,0,0,0};
   float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   float64 a=(float64)((uint64)x.v1 & 0x7fffffff), b=(float64)((uint64)x.v2 & 0x7fffffff);
   float64 maximum=v_f32_reduce_max(v_f32_max_b(a,b));
   // An all-zero group produces zero regardless of scale. Use a nonzero
   // denominator only for that case; positive maxima must not be clamped.
   maximum=v_f32_sel_eq_f32_b(maximum,0.0f,1.0f,maximum);
   // Correct the rounded reciprocal product to the FP32 amax/448 contract.
   // One scale ULP can otherwise move multiple activations across FP8 ties.
   float64 scale=fp8_activation_scale(maximum);
   float64 reciprocal=v_reciprocal_f32(scale);
   minifloat256 packed=fp8_pack_half(x,scale,reciprocal);
   int5 dst={0,row,g,0,0};
   v_f8_st_tnsr_partial(dst,output,packed,127,0);
   v_f32_st_tnsr_partial(dst,scales,scale*2.0f,0,0);
  }
 }
}
