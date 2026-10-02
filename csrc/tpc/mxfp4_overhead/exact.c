#include "exact_math.h"
#include "exact_readers.h"
#ifndef COMPACT
#define COMPACT 0
#endif
#ifndef REPAIR
#define REPAIR 0
#endif
#ifndef DISPATCH
#define DISPATCH 0
#endif
#ifndef BIAS
#define BIAS 0
#endif
/* Compiler-owned scalar local memory, private to a TPC execution context.
 * Globals are required by the TPC C pointer address-space rules. */
__local MxExact mx_sums[2];
__local MxCertificate mx_certificates[2];
__local MxTerm mx_term, mx_bias_terms[2];
static inline unsigned int load_byte(tensor packed,unsigned int offset) {
 int5 p={COMPACT?offset:offset%256,COMPACT?0:offset/256,0,0,0};
 return s_u8_ld_g(gen_addr(p,packed));
}
static inline unsigned int load_scale(tensor scales,unsigned int offset) {
 int5 p={COMPACT?offset:offset%512,COMPACT?0:offset/512,0,0,0};
 return s_u8_ld_g(gen_addr(p,scales));
}
void main(tensor packed,tensor scales,tensor activation,
#if DISPATCH
          tensor partials,tensor flags,
#endif
#if REPAIR
          tensor fast,
#endif
#if BIAS
          tensor bias,
#endif
          tensor output,int N,int K
#if DISPATCH
          ,int splits
#endif
          ) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 /* Standalone exact owns 512 columns; dispatch finish owns 256. Both
  * boundaries retain the native nibble-sharing pair in one task, so splitting
  * finish work never duplicates a logical packed-byte read. */
 for(int row=begin[1];row<end[1];++row)for(int block=begin[0];block<end[0];++block){
#if DISPATCH
  int nblocks=(N+511)/512,task=(row*nblocks+block/2)*splits;
  int5 fp={0,task,0,0,0};unsigned int unsafe=s_u16_ld_g(gen_addr(fp,flags));
  // The predispatch flag covers every split; safe output retains GPU FP32 semantics.
  if(!unsafe && !(N%512) && !(K%32)){
   for(int lane=0;lane<256;lane+=64){
    float64 total=0;
    for(int split=0;split<splits;++split){int5 pp={(task+split)*512+(block%2)*256+lane,0,0,0,0};total+=v_f32_ld_tnsr_b(pp,partials);}
    int5 op={block*256+lane,row,0,0,0};
#if BIAS
    int5 bp={block*256+lane,0,0,0,0};total+=v_f32_ld_tnsr_b(bp,bias);
#endif
    v_f32_st_tnsr(op,output,total);
   }
   continue;
  }
#endif
  int first=block*(DISPATCH?128:256),last=first+(DISPATCH?128:256);
  if(COMPACT && block*512>=N/512*512){first=N/512*256;last=first+N%512;}
  for(int pair=first;pair<last;++pair){
   MxPair columns=mx_pair(pair,N,COMPACT);if(!columns.count)continue;
   unsigned int column[2]={columns.n0,columns.n1},need[2]={1,1},valid[2]={1,1};
   for(int lane=0;lane<columns.count;++lane){
    mx_bias_terms[lane].mantissa=0;mx_bias_terms[lane].shift=0;mx_bias_terms[lane].negative=0;
#if BIAS
    int5 bp={column[lane],0,0,0,0};
    valid[lane]=mx_bias(s_u32_ld_g(gen_addr(bp,bias)),&mx_bias_terms[lane]);
#endif
   }
   /* Exact mode scans once. Diagnostic post-fast repair first classifies,
    * then rereads unsafe outputs; it does not claim one-pass HBM traffic. */
   for(int phase=REPAIR?0:1;phase<=1;++phase){
    unsigned int previous[2]={0xffffffffu,0xffffffffu},cached[2]={0,0},scale[2]={0,0};
    for(int lane=0;lane<columns.count;++lane){
     mx_zero(&mx_sums[lane]);mx_certificate_zero(&mx_certificates[lane]);
     mx_certificates[lane].valid=valid[lane];
    }
    for(int k=0;k<K;++k){
     int5 ap={k,row,0,0,0};unsigned int activation_bits=s_u16_ld_g(gen_addr(ap,activation));
     unsigned int shared_address=0xffffffffu,shared_byte=0;
     for(int lane=0;lane<columns.count;++lane){
      if(!need[lane] || (phase==0 && !mx_certificates[lane].valid))continue;
      MxAddress address=COMPACT?mx_address_v3(column[lane],k,N,K):mx_address_v2(column[lane],k,K);
      if(previous[lane]!=address.byte){
       cached[lane]=address.byte==shared_address?shared_byte:load_byte(packed,address.byte);
       previous[lane]=address.byte;
      }
      shared_address=address.byte;shared_byte=cached[lane];
      if(!(k&31))scale[lane]=load_scale(scales,address.scale);
      unsigned int code=(cached[lane]>>address.shift)&15;
      unsigned int term_valid=mx_product(activation_bits,code,scale[lane],&mx_term);
      if(phase==0){
       mx_certificates[lane].valid=term_valid && scale[lane]>=2 && scale[lane]<=252;
       if(term_valid){int exponent=(activation_bits>>7)&255;
        mx_certificate_term(&mx_certificates[lane],&mx_term,(exponent?exponent:1)-135);}
      }else{valid[lane]&=term_valid;if(term_valid)mx_add(&mx_sums[lane],&mx_term);}
     }
    }
    for(int lane=0;lane<columns.count;++lane){
     if(!need[lane])continue;
     int5 op={column[lane],row,0,0,0};unsigned int result;
#if REPAIR
     if(phase==0){
      mx_certificate_term(&mx_certificates[lane],&mx_bias_terms[lane],-10000);
      if(mx_fp32_safe(&mx_certificates[lane])){
       result=s_u32_ld_g(gen_addr(op,fast));unsigned int magnitude=result&0x7fffffff;
       if(magnitude==0 || (magnitude>=0x00800000 && magnitude<0x7f800000)){
        s_u32_st_g(gen_addr(op,output),result);need[lane]=0;
       }
      }
     }else
#endif
     {
      if(valid[lane]){mx_add(&mx_sums[lane],&mx_bias_terms[lane]);result=mx_round_f32(&mx_sums[lane]);}
      else result=0x7fc00000;
      s_u32_st_g(gen_addr(op,output),result);
     }
    }
    if(!need[0] && (columns.count==1 || !need[1]))break;
   }
  }
 }
}
