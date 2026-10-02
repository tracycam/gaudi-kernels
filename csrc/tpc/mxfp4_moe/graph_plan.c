// Metadata only: no full grouped activation allocation. mode0=CAP=T, mode1=buckets.
void main(tensor ids,tensor counts,tensor inverse,tensor expert_map,int experts,int mode){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int R=get_dim_size(ids,0),T=get_dim_size(ids,1);
 for(int task=b[0];task<end[0];++task){int lower=1,capacity=T,slot_base=0,row_base=0,selected=-1;
  if(mode){int limit=16;while(lower<=T){int slots=s_i32_min(experts,T*R/lower);capacity=s_i32_min(limit,T);if(task<slot_base+slots)break;slot_base+=slots;row_base+=slots*capacity;lower=limit+1;limit*=2;}
   int position=task-slot_base;for(int e=0;e<experts;++e){int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));if(count<lower||count>capacity)continue;if(position==0){selected=e;break;}--position;}
  }else{int5 ep={task,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));if(count>0&&count<=T)selected=task;}
  int5 mp={task,0,0,0,0};s_i32_st_g(gen_addr(mp,expert_map),selected);int row=row_base+(task-slot_base)*capacity;
  if(selected>=0)for(int t=0;t<T;++t)for(int r=0;r<R;++r){int5 rp={r,t,0,0,0};if(s_i32_ld_g(gen_addr(rp,ids))==selected)s_i32_st_g(gen_addr(rp,inverse),row++);}
 }
}
