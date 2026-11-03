// SPDX-License-Identifier: Apache-2.0
// 1<=M<=16384, 256<=H<=8192 (H%256==0). Grid24 repeats small activation statistics, not weights.
// Accumulate the unrounded FP32 residual sum; round residual output and norm to BF16.
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
// Retain the low part of the normalization at BF16 halfway points.
static inline float64 corrected_product(float64 x,float64 inv,float64 inv_lo,float64 weight){
 float64 p=x*inv;
 float64 pe=v_f32_mac_b(x,inv,-p)+x*inv_lo;
 float64 y=p*weight;
 float64 e=v_f32_mac_b(p,weight,-y)+pe*weight;
 uint64 bits=(uint64)y;
 uint64 up=v_u32_sel_eq_u32_b(bits&0x80000000,0,bits+1,bits-1);
 uint64 down=v_u32_sel_eq_u32_b(bits&0x80000000,0,bits-1,bits+1);
 float64 z=v_f32_sel_grt_f32_b(e,0.f,(float64)up,y);
 z=v_f32_sel_less_f32_b(e,0.f,(float64)down,z);
 return v_f32_sel_eq_u32_b(bits&0xffff,0x8000,z,y);
}
void main(tensor x,tensor residual,tensor gamma,tensor residual_out,tensor q_out,tensor scales,float epsilon,int blocks){
 const int width=get_dim_size(x,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[1];row<end[1];++row){
  // Keep a compensated sum and the low product bits until normalization.
  // These bits can determine which side of a BF16 midpoint an output falls on.
  float128 ss={0,0}, correction={0,0};
  int5 pk0={0,row,0,0,0},pk1={128,row,0,0,0};
  // Each task computes the same FP32 sum of squares over the complete row.
  for(int k=0;k<width;k+=256){
   pk0[0]=k; pk1[0]=k+128;
   bfloat128 x0=v_bf16_ld_tnsr_b(pk0,x),r0=v_bf16_ld_tnsr_b(pk0,residual);
   bfloat128 x1=v_bf16_ld_tnsr_b(pk1,x),r1=v_bf16_ld_tnsr_b(pk1,residual);
   float128 xf0=v_convert_bf16_to_f32_all_b(x0), rf0=v_convert_bf16_to_f32_all_b(r0);
   float128 xf1=v_convert_bf16_to_f32_all_b(x1), rf1=v_convert_bf16_to_f32_all_b(r1);
   float128 s0={xf0.v1+rf0.v1,xf0.v2+rf0.v2};
   float128 s1={xf1.v1+rf1.v1,xf1.v2+rf1.v2};
#define ADD_SQUARE(S, V) { \
   float64 term=(V)*(V); \
   float64 term_error=v_f32_mac_b((V),(V),-term); \
   float64 adjusted=term-correction.S; \
   float64 updated=ss.S+adjusted; \
   correction.S=((updated-ss.S)-adjusted)-term_error; \
   ss.S=updated; \
 }
   ADD_SQUARE(v1,s0.v1) ADD_SQUARE(v2,s0.v2)
   ADD_SQUARE(v1,s1.v1) ADD_SQUARE(v2,s1.v2)
#undef ADD_SQUARE
  }
  float64 lanes=ss.v1+ss.v2;
  float64 join1=(ss.v1-lanes)+ss.v2,join2=(ss.v2-lanes)+ss.v1;
  float64 joinerr=v_f32_sel_geq_f32_b(ss.v1,ss.v2,join1,join2);
  float64 total=v_f32_reduce_add(lanes);
  float64 average=total*(1.0f/64.0f);
  float64 sum_residual=v_f32_reduce_add((lanes-average)+(-correction.v1-correction.v2+joinerr));
  float64 sum=total+sum_residual;
  float64 sum_lo=(total-sum)+sum_residual;
  float64 scale=1.0f/(float)width;
  float64 scale_lo=v_f32_mac_b(-scale,(float)width,1.0f)*scale;
  float64 prod=sum*scale;
  float64 prod_lo=v_f32_mac_b(sum,scale,-prod)+sum_lo*scale;
  prod_lo=v_f32_mac_b(sum,scale_lo,prod_lo);
  float64 argument=prod+epsilon;
  float64 argument_lo=((prod-argument)+epsilon)+prod_lo;
  float64 inverse=v_rsqrt_f32(argument);
  float64 square=inverse*inverse;
  float64 square_lo=v_f32_mac_b(inverse,inverse,-square);
  float64 error=v_f32_mac_b(-argument,square,1.0f);
  error=error-(argument*square_lo+argument_lo*square);
  float64 adjust=(0.5f*inverse)*error;
  float64 refined=inverse+adjust;
  float64 inverse_lo=(inverse-refined)+adjust;
  inverse=refined;
  for(int task=begin[0];task<end[0];++task){
   int base_block=task*blocks;
   for(int j=0;j<blocks;j+=2){
    int blk0=base_block+j,blk1=blk0+1;
    int k0=blk0*128,k1=blk1*128;
    int5 p0={k0,row,0,0,0},p1={k1,row,0,0,0},g0={k0,0,0,0,0},g1={k1,0,0,0,0};
    float128 xf0=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0,x));
    float128 xf1=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1,x));
    float128 res0=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0,residual));
    float128 res1=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1,residual));
    float128 rf0={xf0.v1+res0.v1,xf0.v2+res0.v2};
    float128 rf1={xf1.v1+res1.v1,xf1.v2+res1.v2};
    bfloat128 rb0=convert_float128_to_bfloat128(rf0,SW_RHNE);
    bfloat128 rb1=convert_float128_to_bfloat128(rf1,SW_RHNE);
    bfloat128 gw0=v_bf16_ld_tnsr_b(g0,gamma),gw1=v_bf16_ld_tnsr_b(g1,gamma);
    v_bf16_st_tnsr(p0,residual_out,rb0);
    v_bf16_st_tnsr(p1,residual_out,rb1);
    float128 wf0=v_convert_bf16_to_f32_all_b(gw0),wf1=v_convert_bf16_to_f32_all_b(gw1);
    float128 y0={corrected_product(rf0.v1,inverse,inverse_lo,wf0.v1),corrected_product(rf0.v2,inverse,inverse_lo,wf0.v2)};
    float128 y1={corrected_product(rf1.v1,inverse,inverse_lo,wf1.v1),corrected_product(rf1.v2,inverse,inverse_lo,wf1.v2)};
    // This BF16 norm boundary precedes amax and activation conversion.
    bfloat128 norm0=convert_float128_to_bfloat128(y0,SW_RHNE);
    bfloat128 norm1=convert_float128_to_bfloat128(y1,SW_RHNE);
    float64 sc0=scale_from_bits((short128)((ushort128)norm0&0x7fff));
    float64 sc1=scale_from_bits((short128)((ushort128)norm1&0x7fff));
    minifloat256 qv0=quant(norm0,sc0);
    minifloat256 qv1=quant(norm1,sc1);
    int5 q0={0,row,blk0,0,0},q1={0,row,blk1,0,0};
    v_f32_st_tnsr_partial(q0,scales,sc0*2.f,0,0);
    v_f32_st_tnsr_partial(q1,scales,sc1*2.f,0,0);
    v_f8_st_tnsr_partial(q0,q_out,qv0,127,0);
    v_f8_st_tnsr_partial(q1,q_out,qv1,127,0);
   }
  }
 }
}
