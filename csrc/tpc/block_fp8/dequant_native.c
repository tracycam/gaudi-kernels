// Native load/conversion pattern follows 1cat-vllm-gaudi 9f2b52a
// block_fp8_dequant_gaudi2.c (Apache-2.0). Prepared bytes use half-range RNE.
// The two scale policies are compared by full-MAC accuracy, not single values.
#ifndef GK_DECODE_NATIVE_LANES
#define GK_DECODE_NATIVE_LANES 0
#endif
void main(tensor input,tensor scales,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int rows=get_dim_size(input,1);
 for(int nb=begin[1];nb<end[1];++nb) {
  for(int kb=begin[0];kb<end[0];++kb) {
   int5 sp={kb,nb,0,0,0};float scale=s_f32_ld_g(gen_addr(sp,scales));
#if SCALE_BF16
   bfloat128 multiplier=(bf16)scale;
#endif
   #pragma loop_unroll(4)
   for(int r=0;r<128;++r) {
    int row=nb*128+r;bool live=row<rows;
    int5 src={0,row,kb,0,0},dst={kb*128,row,0,0,0};
    minifloat256 packed=v_f8_ld_tnsr_b(src,input,SW_UNPACK|SW_UNPCK_8_TO_16,0,live);
    bfloat128 value=v_convert_f8_to_bf16_b(packed);
#if SCALE_BF16
    bfloat128 result=value*multiplier;
#else
    // The same scalar multiplies every lane; no consumer sees the FP32
    // intermediate. Test the native conversion permutation and its inverse
    // together, avoiding the SW_LINEAR lane rearrangement on both sides.
    float128 wide=convert_bfloat128_to_float128(value,GK_DECODE_NATIVE_LANES?0:SW_LINEAR);
    wide.v1*=scale;wide.v2*=scale;
    bfloat128 result=convert_float128_to_bfloat128(wide,(GK_DECODE_NATIVE_LANES?0:SW_LINEAR)|SW_RHNE);
#endif
    v_bf16_st_tnsr(dst,output,result,0,live);
   }
  }
 }
}
