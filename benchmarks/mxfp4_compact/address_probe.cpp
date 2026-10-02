#include "../../csrc/tpc/mxfp4_compact/address.h"
extern "C" void restore_compact(const unsigned char* w,const unsigned char* s,
                                unsigned N,unsigned K,unsigned char* rows,unsigned char* scales) {
 for(unsigned n=0;n<N;++n) {
  for(unsigned k=0;k<K;++k) {
   unsigned value=(w[compact_weight_offset(n,k,N,K)]>>compact_nibble_shift(n,k,N,K))&15;
   unsigned offset=n*((K+1)/2)+k/2;
   if(k%2)rows[offset]|=value<<4;else rows[offset]=value;
  }
  if(K%2)rows[n*((K+1)/2)+K/2]|=w[compact_weight_offset(n,K-1,N,K)]&240;
  for(unsigned g=0;g<(K+31)/32;++g)scales[n*((K+31)/32)+g]=s[compact_scale_offset(n,g,N,K)];
 }
}
