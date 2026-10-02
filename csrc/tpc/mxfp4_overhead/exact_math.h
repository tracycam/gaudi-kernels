#ifndef GAUDI_MXFP4_EXACT_MATH_H
#define GAUDI_MXFP4_EXACT_MATH_H
/* Pure integer core shared by the TPC slow path and native CPU verification.
 * Every finite BF16*E2M1*E8M0 product is an integer multiple of 2^-261.
 * 18 limbs cover absolute sums for K<=INT_MAX, including finite FP32 bias.
 * The accumulator is signed two's complement; no floating instruction occurs.
 */
#ifdef __TPC__
#define MX_PRIVATE __local
#else
#define MX_PRIVATE
#endif
#define MX_EXACT_LIMBS 18
#define MX_EXACT_QUANTUM 261

typedef struct { unsigned int word[MX_EXACT_LIMBS]; } MxExact;
typedef struct { unsigned int mantissa; int shift; unsigned int negative; } MxTerm;
typedef struct {
 int minimum_bit, maximum_bit, raw_minimum_bit, raw_maximum_bit;
 unsigned int count, valid;
} MxCertificate;

static inline void mx_zero(MxExact MX_PRIVATE *a) {
 for(int i=0;i<MX_EXACT_LIMBS;++i)a->word[i]=0;
}
static inline unsigned int mx_q2(unsigned int code) {
 unsigned int q=code&7;
 return q<4?q:(2+(q&1))<<((q>>1)-1);
}
static inline unsigned int mx_product(unsigned int activation,unsigned int code,
                                      unsigned int scale,MxTerm MX_PRIVATE *term) {
 unsigned int exponent=(activation>>7)&255,mantissa=activation&127;
 if(exponent==255||scale>254)return 0;
 if(exponent)mantissa+=128;else exponent=1;
 term->mantissa=mantissa*mx_q2(code);
 term->shift=(int)exponent+(int)scale-1;
 term->negative=((activation>>15)^(code>>3))&1;
 return 1;
}
static inline unsigned int mx_bias(unsigned int bits,MxTerm MX_PRIVATE *term) {
 unsigned int exponent=(bits>>23)&255,mantissa=bits&0x7fffff;
 if(exponent==255)return 0;
 if(exponent)mantissa+=0x800000;else exponent=1;
 term->mantissa=mantissa;term->shift=(int)exponent+111;
 term->negative=bits>>31;return 1;
}
static inline void mx_add(MxExact MX_PRIVATE *a,const MxTerm MX_PRIVATE *term) {
 if(!term->mantissa)return;
 int first=term->shift>>5,offset=term->shift&31;
 unsigned int lo=term->mantissa<<offset;
 unsigned int hi=offset?term->mantissa>>(32-offset):0,carry=0;
 for(int i=first;i<MX_EXACT_LIMBS;++i){
  unsigned int value=i==first?lo:i==first+1?hi:0,old=a->word[i],next;
  if(term->negative){
   next=old-value;unsigned int b1=old<value,b2=next<carry;
   a->word[i]=next-carry;carry=b1|b2;
  }else{
   next=old+value;unsigned int c1=next<old;
   a->word[i]=next+carry;carry=c1|(a->word[i]<next);
  }
  if(i>=first+1 && !carry)break;
 }
}
static inline int mx_msb(unsigned int value) {
 int bit=-1;while(value){value>>=1;++bit;}return bit;
}
static inline int mx_lsb(unsigned int value) {
 int bit=0;while(!(value&1)){value>>=1;++bit;}return bit;
}
static inline unsigned int mx_bit(const MxExact MX_PRIVATE *a,int bit) {
 return bit<0||bit>=MX_EXACT_LIMBS*32?0:(a->word[bit>>5]>>(bit&31))&1;
}
static inline unsigned int mx_any_below(const MxExact MX_PRIVATE *a,int bit) {
 int whole=bit>>5,offset=bit&31;
 for(int i=0;i<whole;++i)if(a->word[i])return 1;
 return offset&&(a->word[whole]&((1u<<offset)-1));
}
static inline unsigned int mx_shift_low(const MxExact MX_PRIVATE *a,int bit) {
 int first=bit>>5,offset=bit&31;
 unsigned int result=a->word[first]>>offset;
 if(offset&&first+1<MX_EXACT_LIMBS)result|=a->word[first+1]<<(32-offset);
 return result;
}
static inline unsigned int mx_round_f32(MxExact MX_PRIVATE *a) {
 unsigned int sign=a->word[MX_EXACT_LIMBS-1]>>31;
 if(sign){
  unsigned int carry=1;
  for(int i=0;i<MX_EXACT_LIMBS;++i){
   unsigned int value=~a->word[i];a->word[i]=value+carry;
   carry=carry&&(a->word[i]==0);
  }
 }
 int highest=-1;
 for(int i=MX_EXACT_LIMBS-1;i>=0;--i)if(a->word[i]){highest=i*32+mx_msb(a->word[i]);break;}
 if(highest<0)return 0; /* exact zero: sum begins at +0 */
 if(highest>388)return (sign<<31)|0x7f800000u;
 int cut=highest<135?112:highest-23;
 unsigned int mantissa=mx_shift_low(a,cut);
 if(mx_bit(a,cut-1)&&(mx_any_below(a,cut-1)||(mantissa&1)))++mantissa;
 if(highest<135)return (sign<<31)|mantissa; /* includes carry to min-normal */
 if(mantissa==0x1000000){mantissa>>=1;++highest;}
 int exponent=highest-134;
 if(exponent>=255)return (sign<<31)|0x7f800000u;
 return (sign<<31)|((unsigned int)exponent<<23)|(mantissa&0x7fffff);
}

/* A deliberately conservative exactness certificate, only a verification aid.
 * Repair uses mx_fp32_safe below, which allows ordinary FP32 rounding.
 * This exactness check is not a heuristic range
 * check. All products and bias share a normal FP32 quantum, and their absolute
 * sum fits 24 significant bits. Consequently every reduction order is exact.
 * Raw K32 sums are also normal and finite, covering the historical scale-after-
 * MAC path. If the certificate fails, recompute; never accept by absolute error.
 */
static inline void mx_certificate_zero(MxCertificate MX_PRIVATE *c) {
 c->minimum_bit=c->raw_minimum_bit=10000;
 c->maximum_bit=c->raw_maximum_bit=-10000;c->count=0;c->valid=1;
}
static inline void mx_certificate_term(MxCertificate MX_PRIVATE *c,const MxTerm MX_PRIVATE *term,
                                        int raw_shift) {
 if(!term->mantissa)return;
 int low=mx_lsb(term->mantissa),high=mx_msb(term->mantissa);
 int bottom=term->shift-MX_EXACT_QUANTUM+low,top=term->shift-MX_EXACT_QUANTUM+high;
 if(bottom<c->minimum_bit)c->minimum_bit=bottom;
 if(top>c->maximum_bit)c->maximum_bit=top;
 if(raw_shift>-10000){
  if(raw_shift+low<c->raw_minimum_bit)c->raw_minimum_bit=raw_shift+low;
  if(raw_shift+high>c->raw_maximum_bit)c->raw_maximum_bit=raw_shift+high;
 }
 ++c->count;
}
static inline unsigned int mx_certificate_accept(const MxCertificate MX_PRIVATE *c) {
 if(!c->valid)return 0;
 if(!c->count)return 1;
 unsigned int count=c->count-1;int ceil_log2=0;
 while(count){count>>=1;++ceil_log2;}
 return c->minimum_bit>=-126 && c->maximum_bit+ceil_log2<=127 &&
        c->maximum_bit-c->minimum_bit+ceil_log2+1<=24 &&
        c->raw_minimum_bit>=-126 && c->raw_maximum_bit+5<=127;
}
/* GPU-style repair eligibility. Integer lattices prohibit a nonzero subnormal
 * at every operation, even after cancellation. Bounds include the FP32 bias.
 * <=2^20 terms bounds even 3n rounded reduction steps by 3n*u<=3/16
 * (u=2^-24), gamma_3n<=3/13. One exponent of headroom therefore more than
 * covers the rounded absolute sum. Raw K32 MACs
 * have the same margin. Scales 0,1,253,254 never use a BF16-expanded fast path.
 */
static inline unsigned int mx_fp32_safe(const MxCertificate MX_PRIVATE *c) {
 if(!c->valid || c->count>1048576u)return 0;
 if(!c->count)return 1;
 unsigned int count=c->count-1;int ceil_log2=0;
 while(count){count>>=1;++ceil_log2;}
 return c->minimum_bit>=-126 && c->maximum_bit+ceil_log2<=126 &&
        c->raw_minimum_bit>=-126 && c->raw_maximum_bit+5<=126;
}
#endif
