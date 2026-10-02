// Prefix/status/chunk_counts are graph outputs for these same immutable IDs.
void main(tensor ids,tensor prefix,tensor status,tensor chunk_counts,
          tensor row_map,tensor inverse,int rows,int capacity){
 int routes=get_dim_size(ids,0),tokens=get_dim_size(ids,1),length=routes*tokens;
 int experts=get_dim_size(prefix,0)-1,chunks=(length+63)/64;
 int5 z={0,0,0,0,0};int flags=s_i32_ld_g(gen_addr(z,status));
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int task=begin[0];task<end[0];++task){
  if(flags!=0){
   for(int i=task;i<capacity*rows;i+=experts+1){int5 p={i,0,0,0,0};s_i32_st_g(gen_addr(p,row_map),-1);}
   for(int i=task;i<length;i+=experts+1){int5 p={i,0,0,0,0};s_i32_st_g(gen_addr(p,inverse),-1);}
  }else{
   int5 p={task,0,0,0,0};int first=s_i32_ld_g(gen_addr(p,prefix)),last=capacity;
   if(task<experts){p[0]=task+1;last=s_i32_ld_g(gen_addr(p,prefix));}
   if(first<0)first=0;if(first>capacity)first=capacity;if(last<first)last=first;if(last>capacity)last=capacity;
   for(int i=first*rows;i<last*rows;++i){p[0]=i;s_i32_st_g(gen_addr(p,row_map),-1);}
   if(task<experts){
    int ordinal=0;
    for(int chunk=0;chunk<chunks;++chunk){
     int5 cp={chunk,task,0,0,0};int hits=s_i32_ld_g(gen_addr(cp,chunk_counts));
     if(hits>0){
      int first_q=chunk*64,last_q=first_q+64;if(last_q>length)last_q=length;
      for(int q=first_q;q<last_q;++q){
       int5 src={q%routes,q/routes,0,0,0};int id=s_i32_ld_g(gen_addr(src,ids));
       if(id==task){
        int target=first*rows+ordinal++;int5 dst={q,0,0,0,0};
        if(target<last*rows){s_i32_st_g(gen_addr(dst,inverse),target);dst[0]=target;s_i32_st_g(gen_addr(dst,row_map),q);}
        else s_i32_st_g(gen_addr(dst,inverse),-1);
       }
      }
     }
    }
   }
  }
 }
}
