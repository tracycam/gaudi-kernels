// SPDX-License-Identifier: Apache-2.0
// BF16 residual boundary is explicit. Local VLM holds one row (<=8192 BF16).
// No HBM scratch; every x/residual element is loaded once by this kernel.
#ifndef QUANT_POLICY
#define QUANT_POLICY 0
#endif
#if QUANT_POLICY
// Same OCP-RNE then Gaudi-half-RNE adaptation as the separately qualified
// ordinary quantizer. FP32 published scales compensate native-half bytes.
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
#endif
void main(tensor x,tensor residual,tensor gamma,tensor residual_out,tensor norm_out,
#if QUANT_POLICY
 tensor scales,
#endif
 float epsilon){
 const int width=get_dim_size(x,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 bfloat128 saved[64];
 for(int row=begin[0];row<end[0];++row){
  float128 ss={0,0};
#if QUANT_POLICY == 1
  short128 maximum=0;
#endif
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0};
   int count=s_i32_min(width-k,128)-1;
   bfloat128 a=v_bf16_ld_tnsr_partial_b(p,x,(char)count,0);
   bfloat128 b=v_bf16_ld_tnsr_partial_b(p,residual,(char)count,0);
   bfloat128 r=a+b;
   saved[k/128]=r;
   v_bf16_st_tnsr_partial(p,residual_out,r,(char)count,0);
   ss=v_bf16_mac_acc32_b(r,r,ss);
  }
  float64 sum=v_f32_reduce_add(ss.v1+ss.v2);
  float64 inverse=v_rsqrt_f32(sum*(1.0f/(float)width)+epsilon);
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0},g={k,0,0,0,0};
   int count=s_i32_min(width-k,128)-1;
   float128 r=v_convert_bf16_to_f32_all_b(saved[k/128]);
   float128 w=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(g,gamma,(char)count,0));
   float128 y={r.v1*inverse*w.v1,r.v2*inverse*w.v2};
   bfloat128 result=convert_float128_to_bfloat128(y,SW_RHNE);
   #if QUANT_POLICY == 0
   v_bf16_st_tnsr_partial(p,norm_out,result,(char)count,0);
#elif QUANT_POLICY == 1
   saved[k/128]=result;
   maximum=v_i16_max_b(maximum,(short128)((ushort128)result&0x7fff));
#else
   float64 scale=scale_from_bits((short128)((ushort128)result&0x7fff));
   int5 q={0,row,k/128,0,0};
   v_f32_st_tnsr_partial(q,scales,scale*2.f,0,0);
   v_f8_st_tnsr_partial(q,norm_out,quant(result,scale),127,0);
#endif
  }
#if QUANT_POLICY == 1
  float64 scale=scale_from_bits(maximum);
  int5 sp={0,row,0,0,0};v_f32_st_tnsr_partial(sp,scales,scale*2.f,0,0);
  #pragma clang loop unroll_count(4)
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0};int count=s_i32_min(width-k,128)-1;
   v_f8_st_tnsr_partial(p,norm_out,quant(saved[k/128],scale),(char)count,0);
  }
#endif
 }
}
