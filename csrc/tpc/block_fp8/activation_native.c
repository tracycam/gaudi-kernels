// FP8 lane packing follows 1cat-vllm-gaudi 9f2b52a dense_quant_gaudi2.c
// (Apache-2.0); our block128 quantizer, scale policy and subnormal policy differ.
// Original E4M3FN activation contract. One task per row and 128-value block.
// Store exact FP8 values in BF16, or their FP32 dequantization for full-K MME.
#ifndef DEQUANTIZE
#define DEQUANTIZE 0
#endif
static inline float64 quant(float64 x, float64 scale, float64 reciprocal) {
 float64 q=x*reciprocal;
 // A separately rounded q*scale can erase the residual at an FP8 tie.
 // FMA preserves that residual and corrects the approximate reciprocal.
 float64 residual=v_f32_mac_b(-q,scale,x);
 float64 z=v_f32_mac_b(residual,reciprocal,q);
 float64 magnitude=(float64)((uint64)z & 0x7fffffff);
 magnitude=v_f32_min_b(magnitude,448.0f);
 uint64 bits=(uint64)magnitude;
 uint64 rounded=(bits+0x7ffff+((bits>>20)&1)) & 0xfff00000;
 float64 normal=(float64)rounded;
 float64 subnormal=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(magnitude*512.0f,SW_RHNE))*(1.0f/512.0f);
 float64 value=v_f32_sel_less_f32_b(magnitude,0.015625f,subnormal,normal);
 return (float64)((uint64)value | ((uint64)x & 0x80000000));
}
void main(tensor input,tensor output,tensor scales) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[1];row<end[1];++row) {
  for(int g=begin[0];g<end[0];++g) {
   int5 p={g*128,row,0,0,0};
   float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   float64 a=(float64)((uint64)x.v1 & 0x7fffffff), b=(float64)((uint64)x.v2 & 0x7fffffff);
   float64 scale=v_f32_max_b(v_f32_reduce_max(v_f32_max_b(a,b)),1e-10f)*(1.0f/448.0f);
   float64 reciprocal=v_reciprocal_f32(scale);
   float128 q; q.v1=quant(x.v1,scale,reciprocal);q.v2=quant(x.v2,scale,reciprocal);
   bfloat128 bf=convert_float128_to_bfloat128(q,SW_LINEAR|SW_RHNE);
   minifloat256 sparse=v_convert_bf16_to_f8_b(bf*(bf16)0.5f,0,SW_RHNE|SW_CLIP_FP,(minifloat256)0);
   minifloat256 packed=v_f8_pack_b(sparse,SW_GROUP_0|SW_STRIDE_2,(minifloat256)0);
   packed=v_f8_pack_b(sparse,SW_GROUP_1|SW_STRIDE_2,packed);
   packed=v_f8_mov_dual_group_pack_b(packed,SW_PACK21,(minifloat256)0);
   int5 dst={0,row,g,0,0};
   v_f8_st_tnsr_partial(dst,output,packed,127,0);
   v_f32_st_tnsr_partial(dst,scales,scale*2.0f,0,0);
  }
 }
}
