// Split-K FP32 sum folded into the existing BF16 -> FP32 SiLU -> BF16 contract.
void main(tensor partial,tensor counts,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();
#ifdef GK_SMALLM_DENSE
 for(int route=b[1];route<end[1];++route){int5 ep={route,0,0,0,0};int valid=s_i32_ld_g(gen_addr(ep,counts));
  for(int block=b[0];block<end[0];++block){
   int5 dst={block*128,route,0,0,0};
   if(valid<0){v_bf16_st_tnsr(dst,output,(bfloat128)(ushort128)0x7fc0);continue;}
#else
 for(int e=b[2];e<end[2];++e){int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
  for(int row=b[1];row<end[1];++row)for(int block=b[0];block<end[0];++block){
   int5 dst={block*128,row,e,0,0};
   if(row>=count){v_bf16_st_tnsr(dst,output,0);continue;}
#endif
   float128 gs={0},us={0};
   for(int split=0;split<3;++split){
#ifdef GK_SMALLM_DENSE
    int5 gp={block*128,split,route,0,0},up={block*128+256,split,route,0,0};
#else
    int5 gp={block*128,row,split,e,0},up={block*128+256,row,split,e,0};
#endif
    gs.v1+=v_f32_ld_tnsr_b(gp,partial);gp[0]+=64;gs.v2+=v_f32_ld_tnsr_b(gp,partial);
    us.v1+=v_f32_ld_tnsr_b(up,partial);up[0]+=64;us.v2+=v_f32_ld_tnsr_b(up,partial);
   }
   bfloat128 gate=convert_float128_to_bfloat128(gs,SW_LINEAR|SW_RHNE);
   bfloat128 u=convert_float128_to_bfloat128(us,SW_LINEAR|SW_RHNE);
   float128 g=v_convert_bf16_to_f32_all_b(gate);
   g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));
   g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));
   bfloat128 silu=v_convert_f32_to_bf16_all_b(g);
   v_bf16_st_tnsr(dst,output,silu*u);
  }
 }
}
