// Layout v1: [ceil(N/256), K, 128] bytes. Each byte encodes N lane j
// in its low nibble and lane j+128 in its high nibble. E8M0 is
// [ceil(N/256), ceil(K/32), 256]. Input scales must be in [2,252].
// This range makes every nonzero expanded E2M1 value NORMAL, exact BF16.
// Signed zero is preserved by integer exponent adjustment.
static inline bfloat128 scaled(bfloat128 q, ushort128 exponent) {
    ushort128 bits=(ushort128)q,mag=bits&32767;
    ushort128 adjusted=(mag+(((exponent&255)-127)<<7))&32767;
    adjusted=v_u16_sel_eq_u16_b(mag,0,0,adjusted);
    return (bfloat128)(adjusted|(bits&32768));
}
void main(tensor packed,tensor scales,tensor table,tensor previous,tensor output,tensor done,int block_offset) {
    // A real scalar token keeps the intended order in the data graph.
    // Producer writes zero; the initial token is LUT[0], also zero.
    unsigned short prior=s_u16_ld_g(gen_addr((int5){0},previous));
    if(prior!=0)return;
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(output,1);
    for(int nb=begin[0];nb<end[0];++nb) {
      for(int g=begin[1];g<end[1];++g) {
        int5 sp={0,g,nb+block_offset,0,0};
        ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        sp[0]=128;
        ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        for(int k=g*32;k<(g+1)*32 && k<K;++k) {
          int5 p={0,k,nb+block_offset,0,0};
          ushort128 indices=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
          bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
          int5 o={nb*256,k,0,0,0};
          v_bf16_st_tnsr(o,output,scaled(q.v1,s0));o[0]+=128;
          v_bf16_st_tnsr(o,output,scaled(q.v2,s1));
        }
      }
    }
    if(begin[0]==0&&begin[1]==0)
        v_bf16_st_tnsr_partial((int5){0},done,(bfloat128)0,0,0);
}
