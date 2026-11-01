// Route scatters and invalid padding have disjoint write sets. Distribute BOTH
// over the whole index space: assigning all padding of one expert to one task
// leaves most TPCs idle when routes greatly outnumber expert slots.
void main(tensor inverse,tensor valid_rows,tensor status,tensor row_map,int rows){
 int length=get_dim_size(inverse,0),tiles=get_dim_size(valid_rows,0);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status));
 for(int task=begin[0];task<end[0];++task){
  if(!bad&&task<length){int5 p={task,0,0,0,0};int target=s_i32_ld_g(gen_addr(p,inverse));
   if(target>=0&&target<tiles*rows){int5 dst={target,0,0,0,0};s_i32_st_g(gen_addr(dst,row_map),task);}}
  if(task<tiles*rows){int tile=task/rows,row=task%rows;int5 p={tile,0,0,0,0};
   int first=bad?0:s_i32_ld_g(gen_addr(p,valid_rows));
   if(row>=first){int5 dst={task,0,0,0,0};s_i32_st_g(gen_addr(dst,row_map),-1);}
  }
 }
}
