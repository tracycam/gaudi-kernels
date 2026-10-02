// A real join keeps the TPC destination live until the MME branch completes.
void main(tensor src,tensor fence,tensor dst,int rows,int ignored) {
    int5 fp={0,0,0,0,0};float f=s_f32_ld_g(gen_addr(fp,fence));bool ready=f==f;
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int t=begin[0];t<end[0];++t)for(int j=0;j<rows;++j){
        int5 p={0,j,t,0,0};uint64 v=v_u32_ld_tnsr_b(p,src);v_u32_st_tnsr(p,dst,v,0,ready);
    }
}
