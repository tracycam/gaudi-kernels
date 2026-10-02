// Integer metadata only. IDs are [routes,tokens] in TPC order.
void main(tensor ids,tensor counts,tensor row_status,int experts){
 int routes=get_dim_size(ids,0),tokens=get_dim_size(ids,1);
 int5 start=get_index_space_offset(),end=start+get_index_space_size();
 for(int task=start[0];task<end[0];++task){
  if(task<experts){
   int count=0;
   for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r){
    int5 c={r,t,0,0,0};count+=s_i32_ld_g(gen_addr(c,ids))==task;
   }
   int5 c={task,0,0,0,0};s_i32_st_g(gen_addr(c,counts),count);
  }
  if(task<tokens){
   int flags=0;
   for(int r=0;r<routes;++r){
    int5 c={r,task,0,0,0};int id=s_i32_ld_g(gen_addr(c,ids));
    if(id<0||id>=experts)flags|=1;
    for(int j=0;j<r;++j){c[0]=j;if(s_i32_ld_g(gen_addr(c,ids))==id)flags|=2;}
   }
   int5 c={task,0,0,0,0};s_i32_st_g(gen_addr(c,row_status),flags);
  }
 }
}
