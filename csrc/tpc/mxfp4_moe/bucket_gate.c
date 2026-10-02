// Identical BF16/FP32 precision boundaries to the literal production GP gate.
void main(tensor grouped,tensor ids,tensor routing,tensor inverse,tensor status,
          tensor output,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int R=get_dim_size(ids,0),width=get_dim_size(output,0),rows=get_dim_size(grouped,1);
 for(int route=begin[1];route<end[1];++route)for(int block=begin[0];block<end[0];++block){
  int5 rp={route%R,route/R,0,0,0},dst={block*128,route,0,0,0};int expert=s_i32_ld_g(gen_addr(rp,ids));
  if(expert<0||expert>=experts){v_bf16_st_tnsr(dst,output,(bfloat128)0);continue;}
  int5 ep={expert,0,0,0,0};if(s_i32_ld_g(gen_addr(ep,status))){v_bf16_st_tnsr(dst,output,(bfloat128)(ushort128)0x7fc0);continue;}
  int row=s_i32_ld_g(gen_addr(rp,inverse));if(row<0||row>=rows){v_bf16_st_tnsr(dst,output,(bfloat128)(ushort128)0x7fc0);continue;}
  int5 gp={block*128,row,0,0,0},up={block*128+width,row,0,0,0};float128 gs,us;
  gs.v1=v_f32_ld_tnsr_b(gp,grouped);gp[0]+=64;gs.v2=v_f32_ld_tnsr_b(gp,grouped);us.v1=v_f32_ld_tnsr_b(up,grouped);up[0]+=64;us.v2=v_f32_ld_tnsr_b(up,grouped);
  bfloat128 gate=convert_float128_to_bfloat128(gs,SW_LINEAR|SW_RHNE),u=convert_float128_to_bfloat128(us,SW_LINEAR|SW_RHNE);float128 g=v_convert_bf16_to_f32_all_b(gate);
  g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));bfloat128 silu=v_convert_f32_to_bf16_all_b(g);v_bf16_st_tnsr(dst,output,silu*u);
 }
}
