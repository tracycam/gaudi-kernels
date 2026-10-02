// Full64 tensor loads from an explicit graph-owned flattened IDs alias.
void main(tensor ids,tensor flat_ids,tensor counts,tensor row_status,tensor chunk_counts,int experts){
 int routes=get_dim_size(ids,0),tokens=get_dim_size(ids,1),length=routes*tokens;
 int chunks=(length+63)/64;
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int task=begin[0];task<end[0];++task){
  if(task<experts){
   float64 accum=0;int tail_count=0;
   for(int chunk=0;chunk<chunks;++chunk){
    int first=chunk*64;float64 total;
    if(first+64<=length){
     int5 p={first,0,0,0,0};
     int64 packed=v_i32_ld_tnsr_b(p,flat_ids);
     float64 hits=v_f32_sel_eq_i32_b(packed,task,1.0f,0.0f);
     accum+=hits;total=v_f32_reduce_add(hits);
    }else{
     for(int q=first;q<length;++q){int5 p={q%routes,q/routes,0,0,0};tail_count+=s_i32_ld_g(gen_addr(p,ids))==task;}
     total=(float)tail_count;
    }
    int64 integer=convert_float64_to_int64(total,SW_RHNE);int5 dst={chunk,task,0,0,0};
    v_i32_st_tnsr_partial(dst,chunk_counts,integer,0,0);
   }
   float64 total=v_f32_reduce_add(accum)+(float)tail_count;
   int64 integer=convert_float64_to_int64(total,SW_RHNE);int5 dst={task,0,0,0,0};
   v_i32_st_tnsr_partial(dst,counts,integer,0,0);
  }
  if(task<tokens){
   int flags=0;
   for(int r=0;r<routes;++r){
    int5 p={r,task,0,0,0};int id=s_i32_ld_g(gen_addr(p,ids));if(id<0||id>=experts)flags|=1;
    for(int j=0;j<r;++j){p[0]=j;if(s_i32_ld_g(gen_addr(p,ids))==id)flags|=2;}
   }
   int5 dst={task,0,0,0,0};s_i32_st_g(gen_addr(dst,row_status),flags);
  }
 }
}
