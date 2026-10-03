// Only validated selected routes own row-map cells. Padding is not an input
// to counted_gather: it checks valid_rows before dereferencing this tensor.
void main(tensor inverse,tensor valid_rows,tensor status,tensor row_map,int rows){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 if(s_i32_ld_g(gen_addr(z,status)))return;
 int limit=get_dim_size(valid_rows,0)*rows;
 for(int task=b[0];task<end[0];++task){int5 p={task,0,0,0,0};int row=s_i32_ld_g(gen_addr(p,inverse));
  if(row>=0&&row<limit){int5 dst={row,0,0,0,0};s_i32_st_g(gen_addr(dst,row_map),task);}
 }
}
