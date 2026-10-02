void main(tensor partials,tensor output) {
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int splits=get_dim_size(partials,2);
    for(int row=begin[1];row<end[1];++row)for(int block=begin[0];block<end[0];++block){
      float64 total=0;int5 p={block*64,row,0,0,0};
      for(int split=0;split<splits;++split){p[2]=split;total+=v_f32_ld_tnsr_b(p,partials);}
      p[2]=0;v_f32_st_tnsr(p,output,total);
    }
}
