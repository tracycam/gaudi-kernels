// Generic small-M comparator. Every activation row repeats packed weight loads;
// grouped MME is the weight-reusing path. FP32 accumulators AND FP32 stores.
static inline bfloat128 scaled(bfloat128 q, ushort128 exponent) {
    ushort128 bits=(ushort128)q,mag=bits&32767;
    ushort128 adjusted=(mag+(((exponent&255)-127)<<7))&32767;
    adjusted=v_u16_sel_eq_u16_b(mag,0,0,adjusted);
    return (bfloat128)(adjusted|(bits&32768));
}
// MAC_ACC32 returns even lanes in v1 and odd lanes in v2. Restore linear N order
// without any BF16 output conversion or rounding.
static inline float128 linear_acc(float128 x) {
    uint64 lanes=read_lane_id_4b_b();
    // SHUFFLE is local to 16-float dual groups. Bit7 enables each byte;
    // bits0..2 select a float in its group half, bit5 selects the half.
    uchar256 directions=(uchar256)((128+((lanes&15)>>1)+((lanes&16)<<1))*0x01010101);
    bool64 odd=v_u32_cmp_eq_b(lanes&1,1);
    float128 y;
    float64 a=v_f32_mov_dual_group_all_b(x.v1,0xffffffff,0,0,1,1,MkWrA(3,3,3,3),0);
    float64 b=v_f32_mov_dual_group_all_b(x.v2,0xffffffff,0,0,1,1,MkWrA(3,3,3,3),0);
    y.v1=v_f32_shuffle_b(a,directions,0,0);
    y.v1=v_f32_shuffle_vb(b,directions,0,y.v1,odd);
    a=v_f32_mov_dual_group_all_b(x.v1,0xffffffff,2,2,3,3,MkWrA(3,3,3,3),0);
    b=v_f32_mov_dual_group_all_b(x.v2,0xffffffff,2,2,3,3,MkWrA(3,3,3,3),0);
    y.v2=v_f32_shuffle_b(a,directions,0,0);
    y.v2=v_f32_shuffle_vb(b,directions,0,y.v2,odd);
    return y;
}

void main(tensor packed,tensor scales,tensor table,tensor activation,tensor output) {
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(activation,0);
    int splits=get_dim_size(output,2),groups=(K+31)/32,chunk=(groups+splits-1)/splits;
    for(int split=begin[2];split<end[2];++split)
    for(int row=begin[1];row<end[1];++row) for(int nb=begin[0];nb<end[0];++nb) {
        float128 a={0},b={0};
        for(int g=split*chunk;g<(split+1)*chunk && g<groups;++g) {
            int5 sp={0,g,nb,0,0};
            ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]=128;
            ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
            for(int k=g*32;k<(g+1)*32 && k<K;++k) {
                int5 p={0,k,nb,0,0},xp={k,row,0,0,0};
                ushort128 indices=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
                bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
                bf16 x=s_bf16_ld_g(gen_addr(xp,activation));
                a=v_bf16_mac_acc32_b(scaled(q.v1,s0),x,a);
                b=v_bf16_mac_acc32_b(scaled(q.v2,s1),x,b);
            }
        }
        a=linear_acc(a);b=linear_acc(b);
        int5 o={nb*256,row,split,0,0};
        v_f32_st_tnsr(o,output,a.v1);o[0]+=64;v_f32_st_tnsr(o,output,a.v2);
        o[0]+=64;v_f32_st_tnsr(o,output,b.v1);o[0]+=64;v_f32_st_tnsr(o,output,b.v2);
    }
}
