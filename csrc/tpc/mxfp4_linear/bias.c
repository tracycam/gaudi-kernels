void main(tensor x,tensor bias,tensor y) {
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int m=begin[1];m<end[1];++m) for(int nb=begin[0];nb<end[0];++nb) {
      int5 p={nb*64,m,0,0,0},b={nb*64,0,0,0,0};
      v_f32_st_tnsr(p,y,v_f32_ld_tnsr_b(p,x)+v_f32_ld_tnsr_b(b,bias));
    }
}
