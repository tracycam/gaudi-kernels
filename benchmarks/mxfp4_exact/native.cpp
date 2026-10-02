#include "../../csrc/tpc/mxfp4_overhead/exact_math.h"
#include <cstdint>
static_assert(sizeof(unsigned int)==4,"32-bit limbs required");
extern "C" uint32_t exact_dot(const uint16_t*a,const uint8_t*q,const uint8_t*s,
                              int k,uint32_t bias,int use_bias) {
 MxExact sum;mx_zero(&sum);MxTerm term;
 for(int i=0;i<k;++i){if(!mx_product(a[i],q[i],s[i],&term))return 0x7fc00000;mx_add(&sum,&term);}
 if(use_bias){if(!mx_bias(bias,&term))return 0x7fc00000;mx_add(&sum,&term);}
 return mx_round_f32(&sum);
}
extern "C" uint32_t certificate(const uint16_t*a,const uint8_t*q,const uint8_t*s,
                                 int k,uint32_t bias,int use_bias) {
 MxCertificate c;mx_certificate_zero(&c);MxTerm term;
 for(int i=0;i<k;++i){if(!mx_product(a[i],q[i],s[i],&term))return 0;
  if(s[i]<2||s[i]>252)c.valid=0;
  int exponent=(a[i]>>7)&255;mx_certificate_term(&c,&term,(exponent?exponent:1)-135);}
 if(use_bias){if(!mx_bias(bias,&term))return 0;mx_certificate_term(&c,&term,-10000);}
 return mx_certificate_accept(&c);
}

extern "C" uint32_t fp32_safe(const uint16_t*a,const uint8_t*q,const uint8_t*s,
                               int k,uint32_t bias,int use_bias) {
 MxCertificate c;mx_certificate_zero(&c);MxTerm term;
 for(int i=0;i<k;++i){if(!mx_product(a[i],q[i],s[i],&term))return 0;
  if(s[i]<2||s[i]>252)c.valid=0;
  int exponent=(a[i]>>7)&255;mx_certificate_term(&c,&term,(exponent?exponent:1)-135);}
 if(use_bias){if(!mx_bias(bias,&term))return 0;mx_certificate_term(&c,&term,-10000);}
 return mx_fp32_safe(&c);
}
extern "C" uint32_t repeat_cancel(uint16_t activation,uint8_t code,uint8_t scale,
                                  uint32_t count,uint32_t bias) {
 MxExact sum;mx_zero(&sum);MxTerm original,term;
 if(!mx_product(activation,code,scale,&original)||!mx_bias(bias,&term))return 0x7fc00000;
 for(int pass=0;pass<2;++pass){
  for(int bit=0;bit<31;++bit)if(count&(1u<<bit)){
   MxTerm repeated=original;repeated.shift+=bit;repeated.negative^=pass;mx_add(&sum,&repeated);
  }
  if(!pass)mx_add(&sum,&term);
 }
 return mx_round_f32(&sum);
}
#include "../../csrc/tpc/mxfp4_overhead/exact_readers.h"
extern "C" uint32_t reader_v3(int component,uint32_t n,uint32_t k,uint32_t N,uint32_t K) {
 MxAddress a=mx_address_v3(n,k,N,K);return component==0?a.byte:component==1?a.shift:a.scale;
}
extern "C" uint32_t reader_pair(int component,uint32_t pair,uint32_t N,int compact) {
 MxPair p=mx_pair(pair,N,compact);return component==0?p.n0:component==1?p.n1:p.count;
}
