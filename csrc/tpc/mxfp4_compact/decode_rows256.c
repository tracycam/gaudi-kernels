#include "decode_bits.h"
#define ALL_GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
// Compact raw rows in K256 tasks: <=128 original packed bytes and <=8 E8M0
// scale bytes once, with no persistent padding. Only scales2..252 are admitted.
void main(tensor packed,tensor scales,tensor table,tensor output) {
    set_lut_256(table);
    uchar256 lanes=read_lane_id_1b_b();
    uchar256 scale_control=((lanes>>5)<<1)|(lanes&1)|128;
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(output,0);
    for(int n=begin[1];n<end[1];++n) {
      for(int group=begin[0];group<end[0];++group) {
        int count=K-group*256;if(count>256)count=256;
        int bytes=(count+1)/2,groups=(count+31)/32;
        int5 p={group*128,n,0,0,0};
        ushort128 indices=(ushort128)v_u8_ld_tnsr_partial_b(p,packed,bytes-1,0,
            SW_UNPACK|SW_UNPCK_8_TO_16,(uchar256){0});
        int5 s={group*8,n,0,0,0};
        ushort128 compact_exponents=(ushort128)v_u8_ld_tnsr_partial_b(s,scales,groups-1,0,
            SW_UNPACK|SW_UNPCK_8_TO_16,(uchar256){0});
        // Byte SHUFFLE stays within each64B dual group. Replicate the8
        // u16 exponents from dual group0 first, then enable each byte selector.
        uchar256 replicated=v_u8_mov_dual_group_all_b((uchar256)compact_exponents,-1,0,0,0,0,ALL_GROUPS,0);
        ushort128 exponents=(ushort128)v_u8_shuffle_b(replicated,scale_control,0,0);
        bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});
        bfloat128 low=compact_scaled(q.v1,exponents),high=compact_scaled(q.v2,exponents);
        uint128 l=convert_ushort128_to_uint128((ushort128)low,SW_LINEAR);
        uint128 h=convert_ushort128_to_uint128((ushort128)high,SW_LINEAR);
        uint128 first={l.v1,h.v1},second={l.v2,h.v2};
        bfloat128 v0=(bfloat128)v_convert_u32_to_u16_all_b(first);
        bfloat128 v1=(bfloat128)v_convert_u32_to_u16_all_b(second);
        int5 o={group*256,n,0,0,0};int c0=count>128?128:count;
        v_bf16_st_tnsr_partial(o,output,v0,c0-1,0);
        if(count>128){o[0]+=128;v_bf16_st_tnsr_partial(o,output,v1,count-129,0);}
      }
    }
}

#undef ALL_GROUPS
