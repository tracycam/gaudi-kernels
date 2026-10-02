// Fixed cap3 contract. IDs have no role in preparing activations.
void main(tensor activation,tensor counts,tensor packed_a,tensor status) {
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int e=begin[1];e<end[1];++e) {
        int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
        int bad=count<1||count>3;
        if(begin[0]==0)s_i32_st_tnsr_s(ep,status,bad);
        for(int v=begin[0];v<end[0];++v) {
            int5 p0={v*128,0,e,0,0},p1={v*128,1,e,0,0},p2={v*128,2,e,0,0};
            bfloat128 a=v_bf16_ld_tnsr_b(p0,activation,0,(bfloat128)0,count>0&&!bad);
            bfloat128 b=v_bf16_ld_tnsr_b(p1,activation,0,(bfloat128)0,count>1&&!bad);
            bfloat128 c=v_bf16_ld_tnsr_b(p2,activation,0,(bfloat128)0,count>2&&!bad);
            int5 q0={v*128,e*3,0,0,0},q1={v*128,e*3+1,0,0,0},q2={v*128,e*3+2,0,0,0};
            v_bf16_st_tnsr(q0,packed_a,a);v_bf16_st_tnsr(q1,packed_a,b);v_bf16_st_tnsr(q2,packed_a,c);
        }
    }
}
