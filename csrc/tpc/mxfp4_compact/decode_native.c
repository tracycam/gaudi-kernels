#include "decode_bits.h"
// One full N512 block or one N256 half thereof. Views (FCD first):
// packed=[256,KF,NF/512], scales=[512,KF/32,NF/512], output=[NT,KF].
// NT is 256 or 512. pair_offset is 0 for NT512, 0 or 1 for NT256.
// Work index dimension0 enumerates K32 groups. Every original byte in the
// selected N tile is requested once. The graph must not invoke this per M.
// table is the shared 512-BF16 byte-key LUT, 1024 bytes outside weight ledger.
void main(tensor packed,tensor scales,tensor table,tensor output,
          int block_offset,int pair_offset) {
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int pairs=get_dim_size(output,0)/256;
    for(int g=begin[0];g<end[0];++g) {
      for(int pair=0;pair<pairs;++pair) {
        int5 sp={(pair+pair_offset)*256,g,block_offset,0,0};
        ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        sp[0]+=128;
        ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
        for(int k=g*32;k<(g+1)*32;++k) {
          int5 p={(pair+pair_offset)*128,k,block_offset,0,0};
          ushort128 indices=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
          bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
          int5 o={pair*256,k,0,0,0};
          v_bf16_st_tnsr(o,output,compact_native_to_logical(compact_scaled(q.v1,s0)));
          o[0]+=128;
          v_bf16_st_tnsr(o,output,compact_native_to_logical(compact_scaled(q.v2,s1)));
        }
      }
    }
}
