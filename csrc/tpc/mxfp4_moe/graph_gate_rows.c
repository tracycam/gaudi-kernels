// Preserve expert/bucket order; down consumes this BF16 tensor directly.
void main(tensor partial,tensor expert_map,tensor counts,tensor gate_output,int slot_begin){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int width=get_dim_size(gate_output,0);
 for(int local=b[2];local<end[2];++local)for(int row=b[1];row<end[1];++row)for(int block=b[0];block<end[0];++block){
  int5 mp={slot_begin+local,0,0,0,0},dst={block*128,row,local,0,0};int expert=s_i32_ld_g(gen_addr(mp,expert_map));
  if(expert<0){v_bf16_st_tnsr(dst,gate_output,(bfloat128)0);continue;}
  int5 ep={expert,0,0,0,0};if(row>=s_i32_ld_g(gen_addr(ep,counts))){v_bf16_st_tnsr(dst,gate_output,(bfloat128)0);continue;}
  int5 gp={block*128,row,local,0,0},up={block*128+width,row,local,0,0};float128 gs={0},us={0};
  gs.v1+=v_f32_ld_tnsr_b(gp,partial);gp[0]+=64;gs.v2+=v_f32_ld_tnsr_b(gp,partial);us.v1+=v_f32_ld_tnsr_b(up,partial);up[0]+=64;us.v2+=v_f32_ld_tnsr_b(up,partial);
  bfloat128 gate=convert_float128_to_bfloat128(gs,SW_LINEAR|SW_RHNE),u=convert_float128_to_bfloat128(us,SW_LINEAR|SW_RHNE);float128 g=v_convert_bf16_to_f32_all_b(gate);g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));bfloat128 silu=v_convert_f32_to_bf16_all_b(g);v_bf16_st_tnsr(dst,gate_output,silu*u);
 }
}
