// Each output tile owns the ordered FP32 routing reduction. No atomics.
void main(tensor partial,tensor routing,tensor ids,tensor output,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int routes=get_dim_size(routing,0);
 for(int token=begin[1];token<end[1];++token)for(int block=begin[0];block<end[0];++block){
  float64 sum=0;
  for(int slot=0;slot<routes;++slot){
   int5 rp={slot,token,0,0,0};int expert=s_i32_ld_g(gen_addr(rp,ids));
   if(expert<0||expert>=experts)continue;
   float weight=s_f32_ld_g(gen_addr(rp,routing));int5 src={block*64,token*routes+slot,0,0,0};
   sum=v_f32_mac_b(v_f32_ld_tnsr_b(src,partial),weight,sum);
  }
  int5 dst={block*64,token,0,0,0};v_f32_st_tnsr(dst,output,sum);
 }
}
