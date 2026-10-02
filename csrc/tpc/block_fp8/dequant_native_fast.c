// Prepared one-byte FP8 -> scaled BF16 SRAM. Same arithmetic as the baseline.
// Keep eight independent index-register streams; handle only live row blocks.
// Native load/conversion pattern: 1cat-vllm-gaudi 9f2b52a, Apache-2.0.
void main(tensor input,tensor scales,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int rows=get_dim_size(input,1);
 for(int nb=begin[1];nb<end[1];++nb) {
  for(int kb=begin[0];kb<end[0];++kb) {
   int5 sp={kb,nb,0,0,0};float scale=s_f32_ld_g(gen_addr(sp,scales));
   bfloat128 multiplier=(bf16)scale;
   int5 src={0,nb*128,kb,0,0},dst={kb*128,nb*128,0,0,0};
   #pragma loop_unroll(8)
   for(int r=0;r<(rows-nb*128<128?rows-nb*128:128);++r) {
    minifloat256 packed=v_f8_ld_tnsr_b(src,input,SW_UNPACK|SW_UNPCK_8_TO_16,0,1);
    bfloat128 value=v_convert_f8_to_bf16_b(packed);
    bfloat128 result=value*multiplier;
    v_bf16_st_tnsr(dst,output,result);
    src[1]++;dst[1]++;
   }
  }
 }
}
