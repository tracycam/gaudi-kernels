// Input prefix/status must be the outputs of prefix.c for these same IDs.
// Each expert owns disjoint padded tiles; every valid inverse cell has one owner.
void main(tensor ids,tensor prefix,tensor status,tensor row_map,tensor inverse,int rows,int capacity){
 int routes=get_dim_size(ids,0),tokens=get_dim_size(ids,1),experts=get_dim_size(prefix,0)-1;
 int5 z={0,0,0,0,0};int flags=s_i32_ld_g(gen_addr(z,status));
 int5 start=get_index_space_offset(),end=start+get_index_space_size();
 for(int task=start[0];task<end[0];++task){
  if(flags!=0){
   for(int i=task;i<capacity*rows;i+=experts+1){int5 c={i,0,0,0,0};s_i32_st_g(gen_addr(c,row_map),-1);}
   for(int i=task;i<tokens*routes;i+=experts+1){int5 c={i,0,0,0,0};s_i32_st_g(gen_addr(c,inverse),-1);}
  }else{
   int5 c={task,0,0,0,0};int first=s_i32_ld_g(gen_addr(c,prefix)),last=capacity;
   if(task<experts){c[0]=task+1;last=s_i32_ld_g(gen_addr(c,prefix));}
   // Fail safe against out-of-domain externally supplied prefix addresses.
   if(first<0)first=0;if(first>capacity)first=capacity;
   if(last<first)last=first;if(last>capacity)last=capacity;
   for(int i=first*rows;i<last*rows;++i){c[0]=i;s_i32_st_g(gen_addr(c,row_map),-1);}
   if(task<experts){
    int ordinal=0;
    for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r){
     int5 q={r,t,0,0,0};int id=s_i32_ld_g(gen_addr(q,ids));
     if(id==task){
      int route=t*routes+r,target=first*rows+ordinal++;int5 d={route,0,0,0,0};
      if(target<last*rows){s_i32_st_g(gen_addr(d,inverse),target);d[0]=target;s_i32_st_g(gen_addr(d,row_map),route);}
      else s_i32_st_g(gen_addr(d,inverse),-1);
     }
    }
   }
  }
 }
}
