// Layout v1: [ceil(N/256), K, 128] bytes. Each byte encodes N lane j
// in its low nibble and lane j+128 in its high nibble. E8M0 is
// [ceil(N/256), ceil(K/32), 256]. Input scales must be in [2,252].
// This range makes every nonzero expanded E2M1 value NORMAL, exact BF16.
// Signed zero is preserved by integer exponent adjustment.
static inline bfloat128 scaled(bfloat128 q, ushort128 exponent) {
    ushort128 bits=(ushort128)q,mag=bits&32767;
#if defined(GK_MXFP4_DECODE_FAST_SCALE) && GK_MXFP4_DECODE_FAST_SCALE == 2
    // Same validated domain: both operands and every nonzero product are
    // normal BF16 and the E2M1 * power-of-two result is exactly representable.
    // A native BF16 multiply replaces bit masks/add/select without new error.
    // The ld_tnsr return view is not a zero-extension guarantee: mask high
    // bits before making a positive scale. Hardware MUL canonicalizes -0,
    // so predicate zero lanes off and preserve their original income bits.
    bool128 nonzero=v_bf16_cmp_neq_b(q,0);
    return v_bf16_mul_vb(q,(bfloat128)((exponent&255)<<7),0,q,nonzero);
#elif defined(GK_MXFP4_DECODE_FAST_SCALE) && GK_MXFP4_DECODE_FAST_SCALE == 1
    // Validated s=2..252 and nonzero E2M1 q give a BF16 exponent in 1..254.
    // Exponent addition therefore cannot cross the sign bit. For +/-zero,
    // select the original bits, preserving its sign. No floating arithmetic.
    ushort128 adjusted=bits+(((exponent&255)-127)<<7);
    return (bfloat128)v_u16_sel_eq_u16_b(mag,0,bits,adjusted);
#else
    ushort128 adjusted=(mag+(((exponent&255)-127)<<7))&32767;
    adjusted=v_u16_sel_eq_u16_b(mag,0,0,adjusted);
    return (bfloat128)(adjusted|(bits&32768));
#endif
}
void main(tensor packed,tensor scales,tensor table,tensor output,int block_offset) {
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(output,1);
    for(int nb=begin[0];nb<end[0];++nb) {
      for(int g=begin[1];g<end[1];++g) {
        int5 sp={0,g,nb+block_offset,0,0};
        ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        sp[0]=128;
        ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        int k=g*32;
#if defined(GK_MXFP4_DECODE_UNROLL) && GK_MXFP4_DECODE_UNROLL == 4
        // Explicit independent load/lookup chains. The compiler ignored the
        // loop_unroll pragma on the original double-bound loop (identical ISA).
        // Four complete steps plus the scalar-loop tail keep ragged K valid.
        for(;k+3<(g+1)*32 && k+3<K;k+=4) {
          int5 p={0,k,nb+block_offset,0,0};
          ushort128 i0=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
          ushort128 i1=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
          ushort128 i2=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
          ushort128 i3=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
          bfloat256 q0=v_bf16_lookup_2c(i0,0,SW_LUT_PTR,(bfloat256){0});
          bfloat256 q1=v_bf16_lookup_2c(i1,0,SW_LUT_PTR,(bfloat256){0});
          bfloat256 q2=v_bf16_lookup_2c(i2,0,SW_LUT_PTR,(bfloat256){0});
          bfloat256 q3=v_bf16_lookup_2c(i3,0,SW_LUT_PTR,(bfloat256){0});
#define STORE_PAIR(Q,Z) { int5 o={nb*256,Z,0,0,0}; \
 v_bf16_st_tnsr(o,output,scaled(Q.v1,s0));o[0]+=128; \
 v_bf16_st_tnsr(o,output,scaled(Q.v2,s1)); }
          STORE_PAIR(q0,k);STORE_PAIR(q1,k+1);STORE_PAIR(q2,k+2);STORE_PAIR(q3,k+3);
#undef STORE_PAIR
        }
#elif defined(GK_MXFP4_DECODE_UNROLL) && GK_MXFP4_DECODE_UNROLL == 2
        #pragma loop_unroll(2)
#elif defined(GK_MXFP4_DECODE_UNROLL) && GK_MXFP4_DECODE_UNROLL == 8
        #pragma loop_unroll(8)
#endif
        for(;k<(g+1)*32 && k<K;++k) {
          int5 p={0,k,nb+block_offset,0,0};
          ushort128 indices=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
          bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
          int5 o={nb*256,k,0,0,0};
          v_bf16_st_tnsr(o,output,scaled(q.v1,s0));o[0]+=128;
          v_bf16_st_tnsr(o,output,scaled(q.v2,s1));
        }
      }
    }
}
