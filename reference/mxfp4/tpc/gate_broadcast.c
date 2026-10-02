// Fuse gate activation and down replication/map, preserving BF16 boundaries.
void main(tensor partial,tensor ids,tensor copies,tensor mapping,int splits){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();
 int width=get_dim_size(copies,0),slots=get_dim_size(ids,0)*get_dim_size(ids,1),blocks=get_dim_size(copies,1)/slots;
 for(int slot=b[0];slot<end[0];++slot){
  int5 id={slot%get_dim_size(ids,0),slot/get_dim_size(ids,0),0,0,0};
  int expert=s_i32_ld_g(gen_addr(id,ids));
  for(int block=0;block<blocks;++block){int5 mc={slot*blocks+block,0,0,0,0};s_i32_st_g(gen_addr(mc,mapping),expert*blocks+block);}
  for(int row=0;row<width;row+=128){
   int gr=slot*2*width+row,ur=gr+width;
   int gb=(gr/512)*splits*512+gr%512,ub=(ur/512)*splits*512+ur%512;
   float128 gs={0},us={0};
   for(int s=0;s<splits;++s){int5 gp={gb+s*512,0,0,0,0},up={ub+s*512,0,0,0,0};
    gs.v1+=v_f32_ld_tnsr_b(gp,partial);gp[0]+=64;gs.v2+=v_f32_ld_tnsr_b(gp,partial);
    us.v1+=v_f32_ld_tnsr_b(up,partial);up[0]+=64;us.v2+=v_f32_ld_tnsr_b(up,partial);
   }
   bfloat128 gate=v_convert_f32_to_bf16_all_b(gs),up=v_convert_f32_to_bf16_all_b(us);
   float128 g=v_convert_bf16_to_f32_all_b(gate);g.v1=g.v1*v_reciprocal_f32(1.f+v_exp_f32(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+v_exp_f32(-g.v2));
   bfloat128 silu=v_convert_f32_to_bf16_all_b(g),product=silu*up;
   for(int block=0;block<blocks;++block){int5 dst={row,slot*blocks+block,0,0,0};v_bf16_st_tnsr(dst,copies,product);}
  }
 }
}
