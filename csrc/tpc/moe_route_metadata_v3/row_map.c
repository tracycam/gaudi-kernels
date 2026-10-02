// Valid inverse targets and invalid tile padding are disjoint write sets.
// No initialize-then-scatter race, no atomics and no ordering across TPC tasks.
void main(tensor inverse,tensor valid_rows,tensor status,tensor row_map,int rows){
 int length=get_dim_size(inverse,0),tiles=get_dim_size(valid_rows,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size(),zero={0,0,0,0,0};int bad=s_i32_ld_g(gen_addr(zero,status));
 for(int task=begin[0];task<end[0];++task){
  if(!bad&&task<length){int5 p={task,0,0,0,0};int target=s_i32_ld_g(gen_addr(p,inverse));if(target>=0&&target<tiles*rows){int5 dst={target,0,0,0,0};s_i32_st_g(gen_addr(dst,row_map),task);}}
  if(task<tiles){int5 p={task,0,0,0,0};int first=bad?0:s_i32_ld_g(gen_addr(p,valid_rows));for(int row=first;row<rows;++row){int5 dst={task*rows+row,0,0,0,0};s_i32_st_g(gen_addr(dst,row_map),-1);}}
 }
}
