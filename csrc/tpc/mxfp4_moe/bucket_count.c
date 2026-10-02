// One independent task per expert. Invalid IDs do not belong to any expert.
void main(tensor ids,tensor counts,tensor status){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int R=get_dim_size(ids,0),T=get_dim_size(ids,1);
 for(int e=begin[0];e<end[0];++e){int count=0;
  for(int t=0;t<T;++t)for(int r=0;r<R;++r){int5 rp={r,t,0,0,0};count+=s_i32_ld_g(gen_addr(rp,ids))==e;}
  int5 ep={e,0,0,0,0};s_i32_st_g(gen_addr(ep,counts),count);s_i32_st_g(gen_addr(ep,status),count>T);
 }
}
