// Offline candidate, not a qualified HCCL/vendor replacement.
// AG layout is rank-major [8*M,H] FP32. Preserve both BF16 boundaries.
#ifndef VENDOR_BOUNDARIES
#define VENDOR_BOUNDARIES 1
#endif
void main(tensor gathered,tensor residual,tensor gamma,tensor residual_out,tensor norm_out,float epsilon){
 const int width=get_dim_size(residual,0),rows=get_dim_size(residual,1);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 bfloat128 saved[64];
 for(int row=begin[0];row<end[0];++row){
  float128 ss={0,0};
  for(int k=0;k<width;k+=128){
   float128 sum={0,0};
   for(int rank=0;rank<8;++rank){
    int5 p={k,rank*rows+row,0,0,0};
    int first=s_i32_min(width-k,64)-1;
    sum.v1+=v_f32_ld_tnsr_partial_b(p,gathered,(char)first,0);
    if(width-k>64){p[0]+=64;int second=s_i32_min(width-k-64,64)-1;
     sum.v2+=v_f32_ld_tnsr_partial_b(p,gathered,(char)second,0);}
   }
   // Sequential FP32 rank sum from +0, then RNE BF16; not HCCL order proof.
   bfloat128 reduced=convert_float128_to_bfloat128(sum,SW_LINEAR|SW_RHNE);
   int5 p={k,row,0,0,0};int count=s_i32_min(width-k,128)-1;
   bfloat128 r=reduced+v_bf16_ld_tnsr_partial_b(p,residual,(char)count,0);
   saved[k/128]=r;v_bf16_st_tnsr_partial(p,residual_out,r,(char)count,0);
#if VENDOR_BOUNDARIES
   bfloat128 square=r*r;
   float128 wide=v_convert_bf16_to_f32_all_b(square);
   ss.v1+=wide.v1;ss.v2+=wide.v2;
#else
   ss=v_bf16_mac_acc32_b(r,r,ss);
#endif
  }
  float64 total=v_f32_reduce_add(ss.v1+ss.v2);
  float64 inverse=v_rsqrt_f32(total*(1.0f/(float)width)+epsilon);
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0},g={k,0,0,0,0};int count=s_i32_min(width-k,128)-1;
   bfloat128 weight=v_bf16_ld_tnsr_partial_b(g,gamma,(char)count,0);
#if VENDOR_BOUNDARIES
   bfloat128 numerator=saved[k/128]*weight;
   float128 value=v_convert_bf16_to_f32_all_b(numerator);
   value.v1*=inverse;value.v2*=inverse;
#else
   float128 value=v_convert_bf16_to_f32_all_b(saved[k/128]);
   float128 w=v_convert_bf16_to_f32_all_b(weight);
   value.v1=value.v1*inverse*w.v1;value.v2=value.v2*inverse*w.v2;
#endif
   v_bf16_st_tnsr_partial(p,norm_out,convert_float128_to_bfloat128(value,SW_RHNE),(char)count,0);
  }
 }
}
