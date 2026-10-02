// Experimental FP32 Neumaier sum plus FMA product residual. Original MME
// partials and separately rounded FP32 scale factors are preserved. No FP64.
static inline float64 abs32(float64 v){return (float64)((uint64)v&0x7fffffff);}
void main(tensor partial,tensor activation_scales,tensor weight_scales,tensor bias,tensor output){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int groups=get_dim_size(partial,2);
 for(int row=begin[1];row<end[1];++row)for(int nb=begin[0];nb<end[0];++nb){
  int col=nb*128;float64 sumlo=0,sumhi=0,corrlo=0,corrhi=0;
  for(int g=0;g<groups;++g){
   int5 wp={g,nb,0,0,0},ap={0,row,g,0,0},pp={col,row,g,0,0};
   float64 ws=v_f32_ld_g(gen_addr(wp,weight_scales)),as=v_f32_ld_g(gen_addr(ap,activation_scales));float64 factor=ws*as;
   float64 lo=v_f32_ld_tnsr_b(pp,partial);pp[0]+=64;float64 hi=v_f32_ld_tnsr_b(pp,partial);
#define ACCUM(S,V) { \
   float64 term=V*factor; \
   float64 product_error=v_f32_mac_b(V,factor,-term); \
   float64 updated=sum##S+term; \
   float64 first=(sum##S-updated)+term,second=(term-updated)+sum##S; \
   float64 residual=v_f32_sel_geq_f32_b(abs32(sum##S),abs32(term),first,second); \
   corr##S=corr##S+(residual+product_error);sum##S=updated; \
 }
   ACCUM(lo,lo) ACCUM(hi,hi)
#undef ACCUM
  }
  int5 b={col,0,0,0,0};float64 blo=v_f32_ld_tnsr_b(b,bias);b[0]+=64;float64 bhi=v_f32_ld_tnsr_b(b,bias);
  float128 pair={(sumlo+corrlo)+blo,(sumhi+corrhi)+bhi};int5 dst={col,row,0,0,0};v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
 }
}
