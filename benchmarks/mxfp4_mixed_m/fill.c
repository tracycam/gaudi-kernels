// DIAGNOSTIC ONLY: same output geometry/volume, no packed-weight reads.
void main(tensor packed,tensor scales,tensor table,tensor ids,tensor output) {
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int e=begin[2];e<end[2];++e) {
        int5 ep={e,0,0,0,0};int source=s_i32_ld_g(gen_addr(ep,ids));
        ushort128 bits=(source&3)?0x3fc0+(source&3)*0x40:0x3f80;
        bfloat128 one=(bfloat128)bits;
        for(int n=begin[0];n<end[0];++n)for(int g=begin[1];g<end[1];++g) {
            int5 p={n*256,g*32,e,0,0};
            #pragma unroll(8)
            for(int z=0;z<32;++z){v_bf16_st_tnsr(p,output,one);p[0]+=128;v_bf16_st_tnsr(p,output,one);p[0]-=128;p[1]++;}
        }
    }
}
