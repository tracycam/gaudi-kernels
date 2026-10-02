// Native FP8 bytes expand only in a nonpersistent compiler-managed tensor.
// Channel scale stays FP32 in the epilogue; no activation quantization.
#ifndef ROW_BLOCK
#define ROW_BLOCK 8
#endif
void main(tensor input,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int rows=get_dim_size(input,1);
 for(int nb=begin[1];nb<end[1];++nb) {
  for(int kb=begin[0];kb<end[0];++kb) {
   int5 p={kb*128,nb*ROW_BLOCK,0,0,0};
   #pragma loop_unroll(8)
   for(int r=0;r<(rows-nb*ROW_BLOCK<ROW_BLOCK?rows-nb*ROW_BLOCK:ROW_BLOCK);++r) {
    minifloat256 raw=v_f8_ld_tnsr_b(p,input,SW_UNPACK|SW_UNPCK_8_TO_16,0,1);
    v_bf16_st_tnsr(p,output,v_convert_f8_to_bf16_b(raw));p[1]++;
   }
  }
 }
}
