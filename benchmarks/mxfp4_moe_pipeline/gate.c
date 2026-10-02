// Same established precision contract: BF16 gate/up -> FP32 SiLU -> BF16 -> BF16 product.
void main(tensor grouped,tensor ids,tensor counts,tensor output,tensor ready_ids){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int width=get_dim_size(output,0);
 for(int e=b[2];e<end[2];++e){int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
  if(b[0]==0&&b[1]==0)s_i32_st_tnsr_s(ep,ready_ids,s_i32_ld_g(gen_addr(ep,ids)));
  for(int m=b[1];m<end[1];++m)for(int block=b[0];block<end[0];++block){
   int5 dst={block*128,m,e,0,0};if(m>=count){v_bf16_st_tnsr(dst,output,(bfloat128)0);continue;}
   int5 gp={block*128,m,e,0,0},up={block*128+width,m,e,0,0};float128 gs,us;
   gs.v1=v_f32_ld_tnsr_b(gp,grouped);gp[0]+=64;gs.v2=v_f32_ld_tnsr_b(gp,grouped);
   us.v1=v_f32_ld_tnsr_b(up,grouped);up[0]+=64;us.v2=v_f32_ld_tnsr_b(up,grouped);
   bfloat128 gate=convert_float128_to_bfloat128(gs,SW_LINEAR|SW_RHNE),u=convert_float128_to_bfloat128(us,SW_LINEAR|SW_RHNE);
   float128 g=v_convert_bf16_to_f32_all_b(gate);g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));
   bfloat128 silu=v_convert_f32_to_bf16_all_b(g);v_bf16_st_tnsr(dst,output,silu*u);
  }
 }
}
