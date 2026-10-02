#ifndef BIAS
#define BIAS 0
#endif
#if BIAS
void main(tensor partials,tensor bias,tensor output,int nblocks,int splits) {
#else
void main(tensor partials,tensor output,int nblocks,int splits) {
#endif
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[1];row<end[1];++row)for(int block=begin[0];block<end[0];++block){
  int n=block*64,nb=n/512,lane=n%512;float64 total=0;
  for(int split=0;split<splits;++split){int task=(row*nblocks+nb)*splits+split;
   int5 p={task*512+lane,0,0,0,0};total+=v_f32_ld_tnsr_b(p,partials);
  }
  int5 p={n,row,0,0,0};
#if BIAS
  int5 b={n,0,0,0,0};total+=v_f32_ld_tnsr_b(b,bias);
#endif
  v_f32_st_tnsr(p,output,total);
 }
}
