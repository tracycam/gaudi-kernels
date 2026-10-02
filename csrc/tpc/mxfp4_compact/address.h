// MXFP4 compact layout-v3. Identical formula in Python loader and exact reader.
// packed: uint8[N*ceil(K/2)], scales: uint8[N*ceil(K/32)], no tile padding.
// O(1) mapping; arithmetic uses nonnegative logical coordinates.
#ifndef GAUDI_MXFP4_COMPACT_ADDRESS_H
#define GAUDI_MXFP4_COMPACT_ADDRESS_H
static inline unsigned compact_native_lane(unsigned n) {
 return ((n%512)/128)*128+2*(n%64)+(n%128)/64;
}
static inline unsigned compact_weight_offset(unsigned n,unsigned k,unsigned N,unsigned K) {
 unsigned NF=(N/512)*512,KF=(K/32)*32;
 if(k>=KF)return N*(KF/2)+n*((K-KF+1)/2)+(k-KF)/2;
 if(n>=NF)return NF*(KF/2)+(n-NF)*(KF/2)+k/2;
 unsigned p=compact_native_lane(n);
 return (n/512)*KF*256+k*256+(p/256)*128+p%128;
}
static inline unsigned compact_nibble_shift(unsigned n,unsigned k,unsigned N,unsigned K) {
 if(k>=(K/32)*32)return 4*((k-(K/32)*32)%2);
 if(n>=(N/512)*512)return 4*(k%2);
 return 4*((compact_native_lane(n)%256)/128);
}
static inline unsigned compact_scale_offset(unsigned n,unsigned g,unsigned N,unsigned K) {
 unsigned NF=(N/512)*512,G=K/32;
 if(g>=G)return N*G+n;
 if(n>=NF)return NF*G+(n-NF)*G+g;
 return (n/512)*G*512+g*512+compact_native_lane(n);
}
#endif
