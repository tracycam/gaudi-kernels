// SPDX-License-Identifier: Apache-2.0
void main(tensor exponent, tensor sums, tensor result) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int keys = get_dim_size(exponent,0);
  for (int head=begin[0]; head<end[0]; ++head) {
    int5 stat={0,head,0,0,0};
    float64 sum=v_f32_ld_tnsr_b(stat,sums);
    float64 safe=v_f32_sel_grt_f32_b(sum,0.f,sum,1.f);
    float64 inv=v_f32_sel_grt_f32_b(sum,0.f,v_reciprocal_f32(safe),0.f);
    for (int i=0; i<keys; i+=128) {
      int5 p={i,0,head,0,0};
      float64 a=v_f32_ld_tnsr_b(p,exponent)*inv; p[0]+=64;
      float64 b=v_f32_ld_tnsr_b(p,exponent)*inv; p[0]-=64;
      float128 pair={a,b}; v_bf16_st_tnsr(p,result,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
    }
  }
}
