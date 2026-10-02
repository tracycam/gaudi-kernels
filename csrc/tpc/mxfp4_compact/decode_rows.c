#include "decode_bits.h"
// Raw checkpoint row segment; N-tail full KF or K-tail for all N.
// Views (FCD first): packed=[ceil(Kpart/2),Npart], scales=[ceil(Kpart/32),Npart],
// output=[Kpart,Npart]. Kpart is a multiple of32 OR is in1..31.
// Work index dimension0 enumerates K32 groups, dimension1 rows.
// One <=16-byte PARTIAL packed load and one scalar scale load per row/group.
void main(tensor packed,tensor scales,tensor table,tensor output) {
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(output,0);
    for(int n=begin[1];n<end[1];++n) {
      for(int g=begin[0];g<end[0];++g) {
        int count=K-g*32;
        if(count>32)count=32;
        int bytes=(count+1)/2;
        int5 p={g*16,n,0,0,0};
        ushort128 indices=(ushort128)v_u8_ld_tnsr_partial_b(
            p,packed,bytes-1,0,SW_UNPACK|SW_UNPCK_8_TO_16,(uchar256){0});
        int5 s={g,n,0,0,0};
        unsigned char exponent=s_u8_ld_g(gen_addr(s,scales));
        bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
        bfloat128 values=compact_interleave_k(compact_scaled(q.v1,(ushort128)exponent),
                                             compact_scaled(q.v2,(ushort128)exponent));
        int5 o={g*32,n,0,0,0};
        v_bf16_st_tnsr_partial(o,output,values,count-1,0);
      }
    }
}
