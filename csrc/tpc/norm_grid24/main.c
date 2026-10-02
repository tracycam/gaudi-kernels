// SPDX-License-Identifier: Apache-2.0
// M1/H6144 only. Grid24 repeats small activation statistics, not weights.
// Quantization helpers preserve the qualified residual_rmsnorm/fused.c contract.
static inline float64 native_value(float64 x,float64 scale,float64 reciprocal){
 float64 z=x*reciprocal;
 float64 error=v_f32_mac_b(-z,scale,x);
 z=v_f32_mac_b(error,reciprocal,z);
 float64 magnitude=v_f32_min_b((float64)((uint64)z&0x7fffffff),448.f);
 float64 tiny=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(magnitude*512.f,SW_RHNE))*(1.f/1024.f);
 float64 value=v_f32_sel_less_f32_b(magnitude,0.03125f,tiny,magnitude*0.5f);
 return (float64)((uint64)value|((uint64)x&0x80000000));
}
static inline minifloat256 quant(bfloat128 norm,float64 scale){
 float128 x=v_convert_bf16_to_f32_all_b(norm);
 float64 reciprocal=v_reciprocal_f32(scale);
 minifloat256 sparse=0;
 sparse=v_convert_f32_to_f8_b(native_value(x.v1,scale,reciprocal),0,SW_RHNE|SW_CLIP_FP,sparse);
 sparse=v_convert_f32_to_f8_b(native_value(x.v2,scale,reciprocal),2,SW_RHNE|SW_CLIP_FP,sparse);
 minifloat256 packed=v_f8_pack_b(sparse,SW_GROUP_0|SW_STRIDE_2,0);
 packed=v_f8_pack_b(sparse,SW_GROUP_1|SW_STRIDE_2,packed);
 return v_f8_mov_dual_group_pack_b(packed,SW_PACK21,0);
}
static inline float64 scale_from_bits(short128 bits){
 short128 peak=v_i16_reduce_max(bits);
 float128 wide=v_convert_bf16_to_f32_all_b((bfloat128)peak);
 return v_f32_max_b(wide.v1,1e-10f)*(1.f/448.f);
}
void main(tensor x,tensor residual,tensor gamma,tensor residual_out,tensor q_out,tensor scales,float epsilon){
 const int width=get_dim_size(x,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int task=begin[0];task<end[0];++task){
  int start=task*256;
  float128 ss={0,0};
  bfloat128 own0=0,own1=0;
  // Every task repeats exactly the same chronological BF16-add/FP32-MAC chain.
  // The output ownership predicate never changes a statistics operand.
  for(int k=0;k<width;k+=128){
   int5 p={k,0,0,0,0};
   bfloat128 r=v_bf16_ld_tnsr_b(p,x)+v_bf16_ld_tnsr_b(p,residual);
   own0=v_bf16_mov_b(r,0,own0,k==start);
   own1=v_bf16_mov_b(r,0,own1,k==start+128);
   ss=v_bf16_mac_acc32_b(r,r,ss);
  }
  float64 sum=v_f32_reduce_add(ss.v1+ss.v2);
  float64 inverse=v_rsqrt_f32(sum*(1.0f/(float)width)+epsilon);
  for(int j=0;j<2;++j){
   int k=start+j*128;int5 p={k,0,0,0,0};
   bfloat128 saved=v_bf16_mov_b(own1,0,own0,j==1);
   v_bf16_st_tnsr(p,residual_out,saved);
   float128 r=v_convert_bf16_to_f32_all_b(saved);
   float128 w=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p,gamma));
   float128 y={r.v1*inverse*w.v1,r.v2*inverse*w.v2};
   // This BF16 norm boundary precedes amax and activation conversion.
   bfloat128 norm=convert_float128_to_bfloat128(y,SW_RHNE);
   float64 scale=scale_from_bits((short128)((ushort128)norm&0x7fff));
   int5 q={0,0,k/128,0,0};
   v_f32_st_tnsr_partial(q,scales,scale*2.f,0,0);
   v_f8_st_tnsr_partial(q,q_out,quant(norm,scale),127,0);
  }
 }
}
