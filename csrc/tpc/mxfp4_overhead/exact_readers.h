#ifndef GAUDI_MXFP4_EXACT_READERS_H
#define GAUDI_MXFP4_EXACT_READERS_H
/* Byte addressing only. Compact-v3 matches mxfp4_compact/address.h exactly.
 * Callers ensure blob sizes and tensor coordinates fit signed 32-bit indices.
 */
typedef struct { unsigned int byte, shift, scale; } MxAddress;
static inline MxAddress mx_address_v2(unsigned int n,unsigned int k,unsigned int K) {
 unsigned int kp=(K/32+(K%32!=0))*32;
 unsigned int p=(n%512/128)*128+2*(n%64)+(n%128)/64;
 MxAddress result;
 result.byte=(n/512)*kp*256+k*256+(p/256)*128+p%128;
 result.shift=4*((p%256)/128);
 result.scale=(n/512)*(kp/32)*512+(k/32)*512+p;
 return result;
}
static inline MxAddress mx_address_v3(unsigned int n,unsigned int k,unsigned int N,unsigned int K) {
 unsigned int nf=N/512*512,kf=K/32*32,g=kf/32,kt=K-kf;
 MxAddress result;
 if(k>=kf){
  result.byte=N*kf/2+n*((kt+1)/2)+(k-kf)/2;
  result.shift=4*((k-kf)%2);result.scale=N*g+n;
 }else if(n>=nf){
  result.byte=nf*kf/2+(n-nf)*(kf/2)+k/2;
  result.shift=4*(k%2);result.scale=nf*g+(n-nf)*g+k/32;
 }else{
  unsigned int p=(n%512/128)*128+2*(n%64)+(n%128)/64;
  result.byte=(n/512)*kf*256+k*256+(p/256)*128+p%128;
  result.shift=4*((p%256)/128);
  result.scale=(n/512)*g*512+(k/32)*512+p;
 }
 return result;
}
typedef struct {unsigned int n0,n1,count;} MxPair;
static inline MxPair mx_pair(unsigned int pair,unsigned int N,unsigned int compact) {
 unsigned int nf=N/512*512;
 MxPair result;
 if(compact && pair>=nf/2){result.n0=nf+pair-nf/2;result.n1=0;result.count=result.n0<N;}
 else {result.n0=(pair/256)*512+(pair%256/128)*256+pair%128;
       result.n1=result.n0+128;result.count=result.n0>=N?0:result.n1<N?2:1;}
 return result;
}
static inline unsigned int mx_pair_count(unsigned int N,unsigned int compact) {
 return compact?(N/512*512)/2+N%512:(N/512+(N%512!=0))*256;
}
#endif
