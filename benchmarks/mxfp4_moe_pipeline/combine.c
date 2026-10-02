// This fixture routes each token to eight unique experts with weight 1/8.
void main(tensor partial,tensor row_ids,tensor counts,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();int E=get_dim_size(counts,0);
 for(int t=b[1];t<end[1];++t)for(int v=b[0];v<end[0];++v){
  float64 sum=0;
  for(int e=0;e<E;++e){int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));
   for(int m=0;m<count;++m){int5 rp={m,e,0,0,0};int token=s_i32_ld_g(gen_addr(rp,row_ids));int5 p={v*64,m,e,0,0};
    float64 x=v_f32_ld_tnsr_b(p,partial,0,(float64)0,token==t);sum+=x*0.125f;
   }
  }
  int5 q={v*64,t,0,0,0};v_f32_st_tnsr(q,output,sum);
 }
}
