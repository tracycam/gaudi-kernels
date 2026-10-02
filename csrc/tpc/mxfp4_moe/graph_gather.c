// Only this GP fragment's BF16 A is formed. All source values remain unchanged.
void main(tensor input,tensor ids,tensor counts,tensor expert_map,tensor grouped,
          int capacity,int slot_begin){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int K=get_dim_size(input,0),R=get_dim_size(ids,0),T=get_dim_size(ids,1);
 for(int local=b[0];local<end[0];++local){int5 mp={slot_begin+local,0,0,0,0};int expert=s_i32_ld_g(gen_addr(mp,expert_map)),row=0;
  if(expert>=0){int5 ep={expert,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
   if(count<=capacity)for(int t=0;t<T;++t)for(int r=0;r<R;++r){int5 rp={r,t,0,0,0};if(s_i32_ld_g(gen_addr(rp,ids))!=expert)continue;
    for(int k=0;k<K;k+=128){int5 src={k,t,0,0,0},dst={k,row,local,0,0};v_bf16_st_tnsr(dst,grouped,v_bf16_ld_tnsr_b(src,input));}++row;
   }
  }
  for(;row<capacity;++row)for(int k=0;k<K;k+=128){int5 dst={k,row,local,0,0};v_bf16_st_tnsr(dst,grouped,(bfloat128)0);}
 }
}
