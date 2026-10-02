// N-major fused GEMV: original E2M1 nibbles, E8M0 scale, BF16 activation.
// Each task owns 512 output rows. Group partials and final sums use FP32.
#ifndef LOOKUP_SWITCH
#define LOOKUP_SWITCH 0
#endif
#ifndef BLOCKED_LAYOUT
#define BLOCKED_LAYOUT 0
#endif
void main(tensor packed,tensor scales,tensor activation,tensor table,tensor ids,tensor output,int topk_magic)
{
    set_lut_256(table);
    int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int K=256;
    for(int block=start[0];block<end[0];++block){
        int route=block/12, sub=block%12, topk=get_dim_size(ids,0); int5 mi={route%topk,route/topk,0,0,0}; int mapped=s_i32_ld_g(gen_addr(mi,ids))*12+sub;
        int5 w={block*256,0,0,0,0},sc={block*512,0,0,0,0},xc={0,0,0,0,0},oc={block*512,0,0,0,0};
        float128 total0={0},total1={0},total2={0},total3={0};
        for(int group=0;group<K/32;++group){
            float128 part0={0},part1={0},part2={0},part3={0};
            for(int k=group*32;k<(group+1)*32;++k){
                if(BLOCKED_LAYOUT){w[0]=0;w[1]=mapped*K+k;}else{w[1]=k;}
                xc[0]=k;xc[1]=route;
                bf16 x=s_bf16_ld_g(gen_addr(xc,activation));
                uchar256 raw=v_u8_ld_tnsr_b(w,packed);
                ushort256 indices=convert_uchar256_to_ushort256(raw,SW_LINEAR);
                bfloat256 a=v_bf16_lookup_2c(indices.v1,0,SW_LUT_PTR,(bfloat256){0});
                bfloat256 b=v_bf16_lookup_2c(indices.v2,0,SW_LUT_PTR,(bfloat256){0});
                bfloat128 q0=a.v1,q1=a.v2,q2=b.v1,q3=b.v2;
                part0=v_bf16_mac_acc32_b(q0,x,part0);
                part1=v_bf16_mac_acc32_b(q1,x,part1);
                part2=v_bf16_mac_acc32_b(q2,x,part2);
                part3=v_bf16_mac_acc32_b(q3,x,part3);
            }
            sc[0]=BLOCKED_LAYOUT?0:block*512;sc[1]=BLOCKED_LAYOUT?mapped*(K/32)+group:group;
            uchar256 e0=v_u8_ld_tnsr_b(sc,scales);sc[0]+=256;
            uchar256 e1=v_u8_ld_tnsr_b(sc,scales);
            ushort256 s0=convert_uchar256_to_ushort256(e0,SW_LINEAR);
            ushort256 s1=convert_uchar256_to_ushort256(e1,SW_LINEAR);
            ushort128 u0=s0.v1<<7,u1=s0.v2<<7,u2=s1.v1<<7,u3=s1.v2<<7;
            float128 f0=v_convert_bf16_to_f32_all_b((bfloat128)u0);
            float128 f1=v_convert_bf16_to_f32_all_b((bfloat128)u1);
            float128 f2=v_convert_bf16_to_f32_all_b((bfloat128)u2);
            float128 f3=v_convert_bf16_to_f32_all_b((bfloat128)u3);
            total0.v1=v_f32_mac_b(part0.v1,f0.v1,total0.v1);total0.v2=v_f32_mac_b(part0.v2,f0.v2,total0.v2);
            total1.v1=v_f32_mac_b(part1.v1,f1.v1,total1.v1);total1.v2=v_f32_mac_b(part1.v2,f1.v2,total1.v2);
            total2.v1=v_f32_mac_b(part2.v1,f2.v1,total2.v1);total2.v2=v_f32_mac_b(part2.v2,f2.v2,total2.v2);
            total3.v1=v_f32_mac_b(part3.v1,f3.v1,total3.v1);total3.v2=v_f32_mac_b(part3.v2,f3.v2,total3.v2);
        }
        v_bf16_st_tnsr(oc,output,v_convert_f32_to_bf16_all_b(total0));oc[0]+=128;
        v_bf16_st_tnsr(oc,output,v_convert_f32_to_bf16_all_b(total1));oc[0]+=128;
        v_bf16_st_tnsr(oc,output,v_convert_f32_to_bf16_all_b(total2));oc[0]+=128;
        v_bf16_st_tnsr(oc,output,v_convert_f32_to_bf16_all_b(total3));
    }
}
