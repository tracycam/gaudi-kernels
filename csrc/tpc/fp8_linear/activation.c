// Per-token E4M3FN amax/448 RNE, followed by native half-range adaptation.
// FP8 packing derives from the repository's Apache-2.0 block_fp8 kernel.
static inline float64 quant(float64 x,float64 scale,float64 reciprocal) {
 float64 q=x*reciprocal;
 float64 residual=v_f32_mac_b(-q,scale,x);
 float64 z=v_f32_mac_b(residual,reciprocal,q);
 float64 magnitude=v_f32_min_b((float64)((uint64)z&0x7fffffff),448.f);
 uint64 bits=(uint64)magnitude;
 float64 normal=(float64)((bits+0x7ffff+((bits>>20)&1))&0xfff00000);
 float64 sub=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(magnitude*512.f,SW_RHNE))*(1.f/512.f);
 float64 value=v_f32_sel_less_f32_b(magnitude,0.015625f,sub,normal);
 return (float64)((uint64)value|((uint64)x&0x80000000));
}
void main(tensor input,tensor output,tensor scales) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int width=get_dim_size(input,0);
 for(int row=begin[0];row<end[0];++row) {
  float64 maximum=0;
  for(int k=0;k<width;k+=128) {
   int5 p={k,row,0,0,0};float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   maximum=v_f32_max_b(maximum,v_f32_max_b((float64)((uint64)x.v1&0x7fffffff),(float64)((uint64)x.v2&0x7fffffff)));
  }
  float64 scale=v_f32_max_b(v_f32_reduce_max(maximum),1e-10f)*(1.f/448.f);
  float64 reciprocal=v_reciprocal_f32(scale);
  for(int k=0;k<width;k+=128) {
   int5 p={k,row,0,0,0};float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   float128 q={quant(x.v1,scale,reciprocal),quant(x.v2,scale,reciprocal)};
   bfloat128 bf=convert_float128_to_bfloat128(q,SW_LINEAR|SW_RHNE);
   minifloat256 sparse=v_convert_bf16_to_f8_b(bf*(bf16)0.5f,0,SW_RHNE|SW_CLIP_FP,(minifloat256)0);
   minifloat256 packed=v_f8_pack_b(sparse,SW_GROUP_0|SW_STRIDE_2,(minifloat256)0);
   packed=v_f8_pack_b(sparse,SW_GROUP_1|SW_STRIDE_2,packed);
   packed=v_f8_mov_dual_group_pack_b(packed,SW_PACK21,(minifloat256)0);
   v_f8_st_tnsr_partial(p,output,packed,127,0);
  }
  int5 p={0,row,0,0,0};v_f32_st_tnsr_partial(p,scales,scale*2.f,0,0);
 }
}
