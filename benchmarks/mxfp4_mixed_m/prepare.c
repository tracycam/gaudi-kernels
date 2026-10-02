// Dynamic expert counts and IDs remain on device. Capacities describe a cached
// recipe, not observed counts. Each expert owns one unique stable-sort slot.
void main(tensor activation,tensor counts,tensor ids,tensor capacities,
          tensor packed_a,tensor packed_ids,tensor status,int policy) {
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int E=get_dim_size(counts,0),K=get_dim_size(activation,0);
    for(int e=begin[0];e<end[0];++e){
        int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
        int key=policy==2?count:policy==3?(count==3?3:2):0;
        int slot=0;
        for(int j=0;j<E;++j){
            int5 jp={j,0,0,0,0};int c=s_i32_ld_g(gen_addr(jp,counts));
            int k=policy==2?c:policy==3?(c==3?3:2):0;
            slot+=(k<key)||(k==key&&j<e);
        }
        int row0=0;
        for(int j=0;j<slot;++j){int5 jp={j,0,0,0,0};row0+=s_i32_ld_g(gen_addr(jp,capacities));}
        int5 sp={slot,0,0,0,0};int cap=s_i32_ld_g(gen_addr(sp,capacities));
        int bad=(count<1)||(count>cap)||(cap<1)||(cap>3);
        int source=s_i32_ld_g(gen_addr(ep,ids));
        s_i32_st_tnsr_s(sp,packed_ids,source);
        s_i32_st_tnsr_s(ep,status,bad);
        for(int m=0;m<cap;++m)for(int k=0;k<K;k+=128){
            int5 p={k,m,e,0,0},q={k,row0+m,0,0,0};
            bfloat128 v=v_bf16_ld_tnsr_b(p,activation,0,(bfloat128)0,m<count&&!bad);
            v_bf16_st_tnsr(q,packed_a,v);
        }
    }
}
