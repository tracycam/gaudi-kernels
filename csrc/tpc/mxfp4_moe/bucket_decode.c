#include "../mxfp4_compact/decode_bits.h"
#ifndef GK_HISTORICAL_N512
#define GK_HISTORICAL_N512 0
#endif
static inline bfloat128 decoded_lanes(bfloat128 q,ushort128 scale){
 bfloat128 value=compact_scaled(q,scale);
 return GK_HISTORICAL_N512?value:compact_native_to_logical(value);
}
// A mapped expert may repeat across route tiles. Empty slots never read W/S.
void main(tensor packed,tensor scales,tensor table,tensor expert_map,tensor output,
          int nblocks,int slot_begin,int nblock_begin){

#if defined(GK_ROUTE_DECODE_HOIST) && GK_ROUTE_DECODE_HOIST
 set_lut_256(table);int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 const int width_k=get_dim_size(output,1);
 for(int e=begin[2];e<end[2];++e){
  int5 ep={slot_begin+e,0,0,0,0};const int expert=s_i32_ld_g(gen_addr(ep,expert_map));
  for(int block=begin[1];block<end[1];++block){
   const int source=expert*nblocks+nblock_begin+block;
   const int packed_base=GK_HISTORICAL_N512?source*width_k:0;
   const int scale_base=GK_HISTORICAL_N512?source*(width_k/32):0;
   for(int group=begin[0];group<end[0];++group){
    for(int pair=0;pair<2;++pair){ushort128 s0=0,s1=0;
     if(expert>=0){int5 sp={pair*256,scale_base+group,GK_HISTORICAL_N512?0:source,0,0};s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]+=128;s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);}
     int5 wp={pair*128,packed_base+group*32,GK_HISTORICAL_N512?0:source,0,0};
#if defined(GK_ROUTE_DECODE_UNROLL) && GK_ROUTE_DECODE_UNROLL == 2
     #pragma loop_unroll(2)
#elif defined(GK_ROUTE_DECODE_UNROLL) && GK_ROUTE_DECODE_UNROLL == 4
     #pragma loop_unroll(4)
#elif defined(GK_ROUTE_DECODE_UNROLL) && GK_ROUTE_DECODE_UNROLL == 8
     #pragma loop_unroll(8)
#endif
     for(int k=group*32;k<(group+1)*32;++k){bfloat128 first=0,second=0;
      if(expert>=0){ushort128 index=(ushort128)v_u8_ld_tnsr_b(wp,packed,SW_UNPACK|SW_UNPCK_8_TO_16);bfloat256 q=v_bf16_lookup_2c(index,0,SW_LUT_PTR,(bfloat256){0});first=decoded_lanes(q.v1,s0);second=decoded_lanes(q.v2,s1);}
      int5 dst={block*512+pair*256,k,e,0,0};v_bf16_st_tnsr(dst,output,first);dst[0]+=128;v_bf16_st_tnsr(dst,output,second);++wp[1];
     }
    }
   }
  }
 }
#else
 set_lut_256(table);int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int e=begin[2];e<end[2];++e)for(int block=begin[1];block<end[1];++block)for(int group=begin[0];group<end[0];++group){
  int5 ep={slot_begin+e,0,0,0,0};int expert=s_i32_ld_g(gen_addr(ep,expert_map));
  int source=expert*nblocks+nblock_begin+block;
  for(int pair=0;pair<2;++pair){ushort128 s0=0,s1=0;
   if(expert>=0){int5 sp={pair*256,GK_HISTORICAL_N512?source*(get_dim_size(output,1)/32)+group:group,GK_HISTORICAL_N512?0:source,0,0};s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]+=128;s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);}
   for(int k=group*32;k<(group+1)*32;++k){bfloat128 first=0,second=0;
    if(expert>=0){int5 wp={pair*128,GK_HISTORICAL_N512?source*get_dim_size(output,1)+k:k,GK_HISTORICAL_N512?0:source,0,0};ushort128 index=(ushort128)v_u8_ld_tnsr_b(wp,packed,SW_UNPACK|SW_UNPCK_8_TO_16);bfloat256 q=v_bf16_lookup_2c(index,0,SW_LUT_PTR,(bfloat256){0});first=decoded_lanes(q.v1,s0);second=decoded_lanes(q.v2,s1);}
    int5 dst={block*512+pair*256,k,e,0,0};v_bf16_st_tnsr(dst,output,first);dst[0]+=128;v_bf16_st_tnsr(dst,output,second);
   }
  }
 }
#endif
}
