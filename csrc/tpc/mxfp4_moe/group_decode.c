#include "../mxfp4_compact/decode_bits.h"
#ifndef GK_HISTORICAL_N512
#define GK_HISTORICAL_N512 0
#endif
static inline bfloat128 decoded_lanes(bfloat128 q,ushort128 scale){
 bfloat128 value=compact_scaled(q,scale);
 return GK_HISTORICAL_N512?value:compact_native_to_logical(value);
}
// Byte-identical aligned native-v2 payload viewed as [256,K,E*N/512].
// One decode supplies all CAP rows of one expert; empty/overflow groups skip W.
void main(tensor packed,tensor scales,tensor table,tensor counts,tensor output,
          int nblocks,int expert_begin,int nblock_begin,int capacity){
 set_lut_256(table);int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int e=begin[2];e<end[2];++e)for(int block=begin[1];block<end[1];++block)for(int group=begin[0];group<end[0];++group){
  int5 ep={expert_begin+e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
  int source=(expert_begin+e)*nblocks+nblock_begin+block;
  for(int pair=0;pair<2;++pair){ushort128 s0=0,s1=0;
   if(count>0&&count<=capacity){int5 sp={pair*256,GK_HISTORICAL_N512?source*(get_dim_size(output,1)/32)+group:group,GK_HISTORICAL_N512?0:source,0,0};s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]+=128;s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);}
   for(int k=group*32;k<(group+1)*32;++k){bfloat128 first=0,second=0;
    if(count>0&&count<=capacity){int5 wp={pair*128,GK_HISTORICAL_N512?source*get_dim_size(output,1)+k:k,GK_HISTORICAL_N512?0:source,0,0};ushort128 indices=(ushort128)v_u8_ld_tnsr_b(wp,packed,SW_UNPACK|SW_UNPCK_8_TO_16);bfloat256 q=v_bf16_lookup_2c(indices,0,SW_LUT_PTR,(bfloat256){0});first=decoded_lanes(q.v1,s0);second=decoded_lanes(q.v2,s1);}
    int5 dst={block*512+pair*256,k,e,0,0};v_bf16_st_tnsr(dst,output,first);dst[0]+=128;v_bf16_st_tnsr(dst,output,second);
   }
  }
 }
}
