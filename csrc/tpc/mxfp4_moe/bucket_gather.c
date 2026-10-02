// Fixed slots for buckets [1,16],[17,32],... capped at T. Every valid expert
// belongs to one bucket and one deterministic slot (ascending expert ID).
// A and result rows use prefix(sum(slot_count*bucket_capacity)) + local row.
void main(tensor input,tensor ids,tensor counts,tensor grouped,tensor inverse,
          tensor expert_map,int experts,int shared_input){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(input,0),R=get_dim_size(ids,0),T=get_dim_size(ids,1);
 for(int task=begin[0];task<end[0];++task){
  int lower=1,limit=16,slot_base=0,row_base=0,capacity=0;
  while(lower<=T){int slots=s_i32_min(experts,T*R/lower);capacity=s_i32_min(limit,T);
   if(task<slot_base+slots)break;
   slot_base+=slots;row_base+=slots*capacity;lower=limit+1;limit*=2;
  }
  int position=task-slot_base,selected=-1;
  for(int e=0;e<experts;++e){int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
   if(count<lower||count>capacity)continue;
   if(position==0){selected=e;break;}--position;
  }
  int5 mp={task,0,0,0,0};s_i32_st_g(gen_addr(mp,expert_map),selected);
  int first=row_base+(task-slot_base)*capacity,row=0;
  if(selected>=0)for(int t=0;t<T;++t)for(int r=0;r<R;++r){int5 rp={r,t,0,0,0};
   if(s_i32_ld_g(gen_addr(rp,ids))!=selected)continue;
   s_i32_st_g(gen_addr(rp,inverse),first+row);
   for(int k=0;k<K;k+=128){int5 src={k,shared_input?t:t*R+r,0,0,0},dst={k,first+row,0,0,0};v_bf16_st_tnsr(dst,grouped,v_bf16_ld_tnsr_b(src,input));}
   ++row;
  }
  for(;row<capacity;++row)for(int k=0;k<K;k+=128){int5 dst={k,first+row,0,0,0};v_bf16_st_tnsr(dst,grouped,(bfloat128)0);}
 }
}
