// One index task per expert keeps long histogram work balanced at large T.
// Exclusive per-64-ID prefixes form an explicit VPU->memory->SPU graph edge.
void main(tensor ids,tensor flat_ids,tensor counts,tensor row_status,tensor chunk_offsets,int experts){
 int routes=get_dim_size(ids,0),tokens=get_dim_size(ids,1),length=routes*tokens,chunks=(length+63)/64;
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int task=begin[0];task<end[0];++task){
  if(task<experts){
   float64 exclusive=0;
   for(int chunk=0;chunk<chunks;++chunk){
    int5 dst={chunk,task,0,0,0};v_i32_st_tnsr_partial(dst,chunk_offsets,convert_float64_to_int64(exclusive,SW_RHNE),0,0);
    int first=chunk*64;
    if(first+64<=length){int5 src={first,0,0,0,0};int64 packed=v_i32_ld_tnsr_b(src,flat_ids);exclusive+=v_f32_reduce_add(v_f32_sel_eq_i32_b(packed,task,1.f,0.f));}
    else{int tail=0;for(int q=first;q<length;++q){int5 src={q,0,0,0,0};tail+=s_i32_ld_g(gen_addr(src,flat_ids))==task;}exclusive+=(float)tail;}
   }
   int64 total=convert_float64_to_int64(exclusive,SW_RHNE);int5 last={chunks,task,0,0,0},dst={task,0,0,0,0};v_i32_st_tnsr_partial(last,chunk_offsets,total,0,0);v_i32_st_tnsr_partial(dst,counts,total,0,0);
  }
  for(int token=task;token<tokens;token+=experts){
   int flags=0;for(int r=0;r<routes;++r){int5 p={r,token,0,0,0};int id=s_i32_ld_g(gen_addr(p,ids));if(id<0||id>=experts)flags|=1;for(int j=0;j<r;++j){p[0]=j;if(s_i32_ld_g(gen_addr(p,ids))==id)flags|=2;}}
   int5 dst={token,0,0,0,0};s_i32_st_g(gen_addr(dst,row_status),flags);
  }
 }
}
