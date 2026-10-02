#include "../mxfp4_compact/decode_bits.h"
// HistoricalN512 owners; paired BV16 LUT, eight independent K loads/lookups.
// Same N256/K32 issue geometry as the measured native grouped decoder. Empty
// dynamic slots write all zeros but never fetch weight or scale bytes.
void main(tensor packed,tensor scales,tensor table,tensor expert_map,tensor output,
          int nblocks,int slot_begin,int nblock_begin){
 set_lut_256(table);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(output,1);
 for(int e=begin[2];e<end[2];++e){
  int5 ep={slot_begin+e,0,0,0,0};int expert=s_i32_ld_g(gen_addr(ep,expert_map));
  for(int nb=begin[0];nb<end[0];++nb){
   int source=expert*nblocks+nblock_begin+nb/2,pair=nb&1;
   for(int g=begin[1];g<end[1];++g){
    if(expert<0){
     for(int k=g*32;k<(g+1)*32;++k){int5 o={nb*256,k,e,0,0};v_bf16_st_tnsr(o,output,0);o[0]+=128;v_bf16_st_tnsr(o,output,0);}
     continue;
    }
    int5 sp={pair*256,source*(K/32)+g,0,0,0};
    ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]+=128;
    ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
    for(int z=0;z<32;z+=8){
     int k=g*32+z;int5 p={pair*128,source*K+k,0,0,0};
#define LOAD(I) ushort128 i##I=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
     LOAD(0) LOAD(1) LOAD(2) LOAD(3) LOAD(4) LOAD(5) LOAD(6) LOAD(7)
#define LOOKUP(I) bfloat256 q##I=v_bf16_lookup_2c(i##I,0,SW_LUT_PTR,(bfloat256){0});
     LOOKUP(0) LOOKUP(1) LOOKUP(2) LOOKUP(3) LOOKUP(4) LOOKUP(5) LOOKUP(6) LOOKUP(7)
#define STORE(I) {int5 o={nb*256,k+I,e,0,0};v_bf16_st_tnsr(o,output,compact_scaled(q##I.v1,s0));o[0]+=128;v_bf16_st_tnsr(o,output,compact_scaled(q##I.v2,s1));}
     STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5) STORE(6) STORE(7)
    }
   }
  }
 }
}
