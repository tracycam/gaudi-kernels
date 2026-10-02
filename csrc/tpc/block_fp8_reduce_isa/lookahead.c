// Same FP32 factor/product/residual and Neumaier recurrence as the qualified
// source. Only independent preparation of the next group moves earlier.
static inline float64 abs32(float64 v){return (float64)((uint64)v&0x7fffffff);}
#define PREP(G,TLO,THI,ELO,EHI,LIVE) { \
 int5 wp={G,nb,0,0,0},ap={0,row,G,0,0},pp={col,row,G,0,0}; \
 float64 ws=v_f32_ld_g(gen_addr(wp,weight_scales),0,0,LIVE); \
 float64 as=v_f32_ld_g(gen_addr(ap,activation_scales),0,0,LIVE); \
 float64 factor=ws*as; \
 float64 lo=v_f32_ld_tnsr_b(pp,partial,0,0,LIVE);pp[0]+=64; \
 float64 hi=v_f32_ld_tnsr_b(pp,partial,0,0,LIVE); \
 TLO=lo*factor;THI=hi*factor; \
 ELO=v_f32_mac_b(lo,factor,-TLO);EHI=v_f32_mac_b(hi,factor,-THI); \
}
void main(tensor partial,tensor activation_scales,tensor weight_scales,tensor bias,tensor output){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int groups=get_dim_size(partial,2);
 for(int row=begin[1];row<end[1];++row)for(int nb=begin[0];nb<end[0];++nb){
  int col=nb*128;float64 sumlo=0,sumhi=0,corrlo=0,corrhi=0;
  float64 termlo,termhi,errorlo,errorhi;
  PREP(0,termlo,termhi,errorlo,errorhi,1)
  for(int g=0;g<groups;++g){
   float64 nextlo,nexthi,nexterrlo,nexterrhi;bool live=g+1<groups;
   PREP(g+1,nextlo,nexthi,nexterrlo,nexterrhi,live)
#define ACCUM(S) { \
   float64 updated=sum##S+term##S; \
   float64 first=(sum##S-updated)+term##S,second=(term##S-updated)+sum##S; \
   float64 residual=v_f32_sel_geq_f32_b(abs32(sum##S),abs32(term##S),first,second); \
   corr##S=corr##S+(residual+error##S);sum##S=updated; \
 }
   ACCUM(lo) ACCUM(hi)
#undef ACCUM
   termlo=nextlo;termhi=nexthi;errorlo=nexterrlo;errorhi=nexterrhi;
  }
  int5 b={col,0,0,0,0};float64 blo=v_f32_ld_tnsr_b(b,bias);b[0]+=64;float64 bhi=v_f32_ld_tnsr_b(b,bias);
  float128 pair={(sumlo+corrlo)+blo,(sumhi+corrhi)+bhi};int5 dst={col,row,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
 }
}
