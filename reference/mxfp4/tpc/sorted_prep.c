// Device-side expert selection and BF16 activation replication.
void main(tensor x,tensor ids,tensor permutation,tensor copies,tensor mapping,tensor inverse,int blocks,int splits,int topk){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int kp=get_dim_size(copies,0);
 for(int task=begin[0];task<end[0];++task){
  int slot=task/(blocks*splits),sub=task%(blocks*splits);
  int5 ic={slot%topk,slot/topk,0,0,0};
  int expert=s_i32_ld_g(gen_addr(ic,ids));
  int original=s_i32_ld_g(gen_addr(ic,permutation));
  if(sub==0){int5 inv={original%topk,original/topk,0,0,0};s_i32_st_g(gen_addr(inv,inverse),slot);}
  int mapped=expert*blocks*splits+sub;
  int5 mc={task,0,0,0,0};
  s_i32_st_g(gen_addr(mc,mapping),mapped);
  for(int v=0;v<kp;v+=128){
   int5 src={(sub%splits)*kp+v,original/topk,0,0,0},dst={v,task,0,0,0};
   v_bf16_st_tnsr(dst,copies,v_bf16_ld_tnsr_b(src,x));
  }
 }
}
