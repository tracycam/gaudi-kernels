// SPDX-License-Identifier: Apache-2.0
void main(tensor x,tensor residual,tensor gamma,tensor residual_out,tensor norm_out,
          float epsilon){
 const int width=get_dim_size(x,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[0];row<end[0];++row){
  float128 ss={0,0};
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0};int count=s_i32_min(width-k,128)-1;
   float128 a=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(p,x,(char)count,0),SW_LINEAR);
   float128 b=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(p,residual,(char)count,0),SW_LINEAR);
   float128 r={a.v1+b.v1,a.v2+b.v2};
   v_bf16_st_tnsr_partial(p,residual_out,convert_float128_to_bfloat128(r,SW_RHNE|SW_LINEAR),(char)count,0);
   ss.v1=v_f32_mac_b(r.v1,r.v1,ss.v1);ss.v2=v_f32_mac_b(r.v2,r.v2,ss.v2);
  }
  float64 sum=v_f32_reduce_add(ss.v1+ss.v2);
  float64 inverse=v_rsqrt_f32(sum*(1.0f/(float)width)+epsilon);
  for(int k=0;k<width;k+=128){
   int5 p={k,row,0,0,0},g={k,0,0,0,0};int count=s_i32_min(width-k,128)-1;
   float128 a=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(p,x,(char)count,0),SW_LINEAR);
   float128 b=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(p,residual,(char)count,0),SW_LINEAR);
   float128 r={a.v1+b.v1,a.v2+b.v2};
   float128 w=convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(g,gamma,(char)count,0),SW_LINEAR);
   float128 y={r.v1*inverse*w.v1,r.v2*inverse*w.v2};
   v_f32_st_tnsr(p,norm_out,y.v1);
   p[0]+=64;v_f32_st_tnsr(p,norm_out,y.v2);
  }
 }
}
