// Per-token dynamic OCP FP8 activation quantization.
#include "activation_quant_common.h"

void main(tensor input, tensor output, tensor scales) {
 int5 begin=get_index_space_offset(), end=begin+get_index_space_size();
 int width=get_dim_size(input,0);
 for (int row=begin[0]; row<end[0]; ++row) {
  float64 maximum=0.0f;
  for (int col=0; col<width; col+=128) {
   int5 p={col,row,0,0,0};
   float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   float64 a=(float64)((uint64)x.v1 & 0x7fffffff), b=(float64)((uint64)x.v2 & 0x7fffffff);
   maximum=v_f32_max_b(maximum,v_f32_reduce_max(v_f32_max_b(a,b)));
  }
  // Only an all-zero token needs a substitute denominator.
  maximum=v_f32_sel_eq_f32_b(maximum,0.0f,1.0f,maximum);
  float64 scale=fp8_activation_scale(maximum), reciprocal=v_reciprocal_f32(scale);
  for (int col=0; col<width; col+=128) {
   int5 p={col,row,0,0,0};
   float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,input),SW_LINEAR);
   v_f8_st_tnsr_partial(p,output,fp8_pack_half(x,scale,reciprocal),127,0);
  }
  int5 s={0,row,0,0,0};
  v_f32_st_tnsr_partial(s,scales,scale*2.0f,0,0);
 }
}
