void main(tensor input,tensor output,float increment){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();
 for(int task=b[0];task<end[0];++task){
  int5 p={task*64,0,0,0,0};float64 x=v_f32_ld_tnsr_b(p,input);
  v_f32_st_tnsr(p,output,x+increment);
 }
}
