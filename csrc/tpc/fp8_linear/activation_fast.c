// Exact ordinary E4M3FN policy, with native conversion for the normal range.
// The all_b expansion gives interleaved even/odd BF16 lanes for quarter0/2.
// FP32 quarter packing follows 1cat-vllm-gaudi9f2b52a,Apache-2.0.
// Unlike that reference's power-of-two quantizer, retain amax/448 and subnormals.
static inline float64 native_value(float64 x,float64 scale,float64 reciprocal) {
 float64 q=x*reciprocal;
 float64 residual=v_f32_mac_b(-q,scale,x);
 float64 z=v_f32_mac_b(residual,reciprocal,q);
 float64 mag=v_f32_min_b((float64)((uint64)z&0x7fffffff),448.f);
 // For original magnitudes below1/32, OCP uses a1/512 grid. Round there
 // before native half-range conversion to preserve the two RNE operations.
 float64 tiny=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(mag*512.f,SW_RHNE))*(1.f/1024.f);
 float64 value=v_f32_sel_less_f32_b(mag,0.03125f,tiny,mag*0.5f);
 return (float64)((uint64)value|((uint64)x&0x80000000));
}
void main(tensor input,tensor output,tensor scales) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int width=get_dim_size(input,0);
 for(int row=begin[0];row<end[0];++row) {
  float64 maximum=0;
  #pragma loop_unroll(4)
  for(int k=0;k<width;k+=128) {
   int5 p={k,row,0,0,0};float128 x=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p,input));
   maximum=v_f32_max_b(maximum,v_f32_max_b((float64)((uint64)x.v1&0x7fffffff),(float64)((uint64)x.v2&0x7fffffff)));
  }
  float64 scale=v_f32_max_b(v_f32_reduce_max(maximum),1e-10f)*(1.f/448.f);
  float64 reciprocal=v_reciprocal_f32(scale);
  #pragma loop_unroll(4)
  for(int k=0;k<width;k+=128) {
   int5 p={k,row,0,0,0};float128 x=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p,input));
   minifloat256 sparse=0;
   sparse=v_convert_f32_to_f8_b(native_value(x.v1,scale,reciprocal),0,SW_RHNE|SW_CLIP_FP,sparse);
   sparse=v_convert_f32_to_f8_b(native_value(x.v2,scale,reciprocal),2,SW_RHNE|SW_CLIP_FP,sparse);
   minifloat256 packed=v_f8_pack_b(sparse,SW_GROUP_0|SW_STRIDE_2,(minifloat256)0);
   packed=v_f8_pack_b(sparse,SW_GROUP_1|SW_STRIDE_2,packed);
   packed=v_f8_mov_dual_group_pack_b(packed,SW_PACK21,(minifloat256)0);
   v_f8_st_tnsr_partial(p,output,packed,127,0);
  }
  int5 p={0,row,0,0,0};v_f32_st_tnsr_partial(p,scales,scale*2.f,0,0);
 }
}
