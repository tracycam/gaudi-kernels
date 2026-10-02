// Literal precision contract: round gate/up to BF16; evaluate SiLU in FP32;
// round SiLU to BF16; BF16 multiply with up. Grouped MME rows are logical-N,
// so the first narrowing is explicitly linear, unlike native MAC partial lanes.
void main(tensor grouped,tensor ids,tensor routing,tensor inverse,tensor overflow,
          tensor output,int capacity,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int R=get_dim_size(routing,0),width=get_dim_size(output,0);
 for(int route=begin[1];route<end[1];++route)for(int block=begin[0];block<end[0];++block){
  int5 rp={route%R,route/R,0,0,0},dst={block*128,route,0,0,0};int e=s_i32_ld_g(gen_addr(rp,ids));
  if(e<0||e>=experts){v_bf16_st_tnsr(dst,output,(bfloat128)0);continue;}
  int5 ep={e,0,0,0,0};int bad=s_i32_ld_g(gen_addr(ep,overflow)),row=s_i32_ld_g(gen_addr(rp,inverse));
  if(bad||row<0||row>=capacity){v_bf16_st_tnsr(dst,output,(bfloat128)(ushort128)0x7fc0);continue;}
  int5 gp={block*128,row,e,0,0},up={block*128+width,row,e,0,0};float128 gs,us;
  gs.v1=v_f32_ld_tnsr_b(gp,grouped);gp[0]+=64;gs.v2=v_f32_ld_tnsr_b(gp,grouped);
  us.v1=v_f32_ld_tnsr_b(up,grouped);up[0]+=64;us.v2=v_f32_ld_tnsr_b(up,grouped);
  bfloat128 gate=convert_float128_to_bfloat128(gs,SW_LINEAR|SW_RHNE),u=convert_float128_to_bfloat128(us,SW_LINEAR|SW_RHNE);
  float128 g=v_convert_bf16_to_f32_all_b(gate);
  g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));
  bfloat128 silu=v_convert_f32_to_bf16_all_b(g);v_bf16_st_tnsr(dst,output,silu*u);
 }
}
