// Bounded finite-weight diagnostic. The three decoded tensors are true inputs,
// so all three must be materialized before any gated activation can be consumed.
// Checking one finite BF16 pair as raw FP32 bits preserves the activation bytes.
void main(tensor activation,tensor d0,tensor d1,tensor d2,tensor output) {
    int5 p={0,0,0,0,0};
    unsigned a=s_u32_ld_g(gen_addr(p,d0));
    unsigned b=s_u32_ld_g(gen_addr(p,d1));
    unsigned c=s_u32_ld_g(gen_addr(p,d2));
    bool ready=((a&0x7f800000)!=0x7f800000)&&
               ((b&0x7f800000)!=0x7f800000)&&
               ((c&0x7f800000)!=0x7f800000);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int e=begin[2];e<end[2];++e)
      for(int m=begin[1];m<end[1];++m)
        for(int k=begin[0];k<end[0];++k){
            int5 q={k*128,m,e,0,0};
            bfloat128 v=v_bf16_ld_tnsr_b(q,activation);
            v_bf16_st_tnsr(q,output,v,0,ready);
        }
}
