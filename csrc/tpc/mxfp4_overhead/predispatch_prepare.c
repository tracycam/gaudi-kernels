#include "exact_math.h"
#ifndef BIAS
#define BIAS 0
#endif
/* Load-time min/max scale metadata replaces a runtime reread of weight scales.
 * One index owns all splits of a whole Nblock/M row: its guard is broadcast. */
void main(tensor input,
#if BIAS
          tensor bias,
#endif
          tensor activation,tensor unsafe,int N,int K,int nblocks,int splits,
          int scale_min,int scale_max,int force_exact) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int kp=get_dim_size(activation,0);ushort128 lanes=read_lane_id_2b_b();
 for(int entry=begin[0];entry<end[0];++entry){
  int row=entry/nblocks,nb=entry%nblocks;
  ushort128 minimum=255,maximum=0,invalid=0;
  for(int split=0;split<splits;++split)for(int k=0;k<kp;k+=128){
   int start=split*kp+k;int5 src={start,row,0,0,0},dst={k,entry*splits+split,0,0,0};
   ushort128 bits=(ushort128)v_bf16_ld_tnsr_b(src,input);
   int remaining=K-start<kp-k?K-start:kp-k;
   if(remaining>128)remaining=128; // avoid ushort truncation at Kpart >= 65536
   bits=v_u16_sel_geq_u16_b(lanes,(unsigned short)remaining,0,bits,0,bits,remaining>0);
   if(start>=K)bits=0;
   v_bf16_st_tnsr(dst,activation,(bfloat128)bits);
   ushort128 magnitude=bits&32767,exponent=(magnitude>>7)&255;
   invalid|=v_u16_sel_eq_u16_b(exponent,255,1,0);
   exponent=v_u16_sel_eq_u16_b(exponent,0,1,exponent);
   ushort128 lo=v_u16_sel_eq_u16_b(magnitude,0,255,exponent);
   ushort128 hi=v_u16_sel_eq_u16_b(magnitude,0,0,exponent);
   minimum=v_u16_min_b(minimum,lo);maximum=v_u16_max_b(maximum,hi);
  }
  short128 min_reduced=v_i16_reduce_min((short128)minimum);
  short128 max_reduced=v_i16_reduce_max((short128)maximum);
  short128 invalid_reduced=v_i16_reduce_max((short128)invalid);
  unsigned int bad=force_exact || N%512 || K%32 ||
                   scale_min<2 || scale_max>252 || scale_min>scale_max;
  unsigned int count=(unsigned int)K+1;int ceil_log2=0;
  for(unsigned int remaining=count-1;remaining;remaining>>=1)++ceil_log2;
  if(count>1048576u)bad=1;
  short128 lowest=min_reduced+(short)(scale_min-262);
  short128 highest=max_reduced+(short)(scale_max-251);
  lowest=v_i16_sel_eq_i16_b(max_reduced,0,1000,lowest);
  highest=v_i16_sel_eq_i16_b(max_reduced,0,-1000,highest);
  short128 raw_bad=v_i16_sel_less_i16_b(min_reduced,(short)9,1,0);
  raw_bad|=v_i16_sel_grt_i16_b(max_reduced,(short)245,1,0);
  raw_bad=v_i16_sel_eq_i16_b(max_reduced,0,0,raw_bad);
#if BIAS
  // Include every significand bit in the bound. This conservative exponent-only
  // scan is vectorized and may send small normal biases to exact unnecessarily;
  // it can never admit a lattice or sum range rejected by the tighter scan.
  ushort128 bias_minimum=255,bias_maximum=0,bias_invalid=0;
  for(int col=nb*512;col<(nb+1)*512;col+=64){
   int5 bp={col,0,0,0,0};uint64 bits=v_u32_ld_tnsr_b(bp,bias);
   uint64 magnitude=bits&0x7fffffff,exponent=(bits>>23)&255;
   bias_invalid|=(ushort128)v_u32_sel_eq_u32_b(exponent,255,1,0);
   exponent=v_u32_sel_eq_u32_b(exponent,0,1,exponent);
   ushort128 lo=(ushort128)v_u32_sel_eq_u32_b(magnitude,0,255,exponent);
   ushort128 hi=(ushort128)v_u32_sel_eq_u32_b(magnitude,0,0,exponent);
   // uint32 -> uint16 reinterpretation inserts one zero high half per lane.
   lo=v_u16_sel_eq_u16_b(lanes&1,1,255,lo);
   bias_minimum=v_u16_min_b(bias_minimum,lo);
   bias_maximum=v_u16_max_b(bias_maximum,hi);
  }
  short128 bias_max=v_i16_reduce_max((short128)bias_maximum);
  short128 bias_low=v_i16_reduce_min((short128)bias_minimum)-150;
  short128 bias_high=bias_max-127;
  bias_low=v_i16_sel_eq_i16_b(bias_max,0,1000,bias_low);
  bias_high=v_i16_sel_eq_i16_b(bias_max,0,-1000,bias_high);
  lowest=v_i16_min_b(lowest,bias_low);highest=v_i16_max_b(highest,bias_high);
  invalid_reduced|=v_i16_reduce_max((short128)bias_invalid);
#endif
  short128 vector_bad=raw_bad|invalid_reduced|(short)bad;
  vector_bad|=v_i16_sel_less_i16_b(lowest,(short)-126,1,0);
  vector_bad|=v_i16_sel_grt_i16_b(highest,(short)(126-ceil_log2),1,0);
  for(int split=0;split<splits;++split){int5 f={0,entry*splits+split,0,0,0};v_u16_st_tnsr_partial(f,unsafe,(ushort128)vector_bad,0,0);}
 }
}
