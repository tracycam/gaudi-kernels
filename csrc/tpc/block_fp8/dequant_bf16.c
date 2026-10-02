#define PACKED 1
// Original OCP byte + original FP32 block scale -> transient BF16 for MME.
// One scalar scale load per 128x128 weight tile, never per weight vector.
static inline bfloat128 exact(uchar256 packed) {
 ushort128 raw=(ushort128)packed,mag=raw&127;
 ushort128 normal=(mag<<4)+0x3c00;
 bfloat128 subnormal=v_convert_u16_to_bf16_b(mag)*(bf16)(1.0f/512.0f);
 bfloat128 value=v_bf16_sel_less_u16_b(mag,8,subnormal,(bfloat128)normal);
 ushort128 bits=v_u16_sel_eq_u16_b(mag,127,0x7fc0,(ushort128)value);
 return (bfloat128)(bits|((raw&128)<<8));
}
void main(tensor input,tensor scales,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int rows=get_dim_size(input,1);
 for(int nb=begin[1];nb<end[1];++nb) {
  for(int kb=begin[0];kb<end[0];++kb) {
   int5 sp={kb,nb,0,0,0};float scale=s_f32_ld_g(gen_addr(sp,scales));
   #pragma loop_unroll(4)
   for(int r=0;r<128;++r) {
    int row=nb*128+r;
    bool live=row<rows;
#ifdef PACKED
     int5 src={0,row,kb,0,0};
#else
     int5 src={kb*128,row,0,0,0};
#endif
     int5 p={kb*128,row,0,0,0};
     bfloat128 decoded=exact(v_u8_ld_tnsr_b(src,input,SW_UNPACK|SW_UNPCK_8_TO_16,0,live));
     float128 f=convert_bfloat128_to_float128(decoded,SW_LINEAR);
     f.v1*=scale;f.v2*=scale;
     v_bf16_st_tnsr(p,output,convert_float128_to_bfloat128(f,SW_LINEAR|SW_RHNE),0,live);
   }
  }
 }
}
