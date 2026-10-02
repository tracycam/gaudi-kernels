// Real token gather, included in both upstream comparisons.
// Separate path writes NaN to inactive slots, then lean preparation masks them.
// Fused path directly writes the final flat, zero-padded activation layout.
void main(tensor tokens,tensor row_ids,tensor counts,tensor output,tensor status,int flat) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int T=get_dim_size(tokens,1);
 for(int e=begin[1];e<end[1];++e){
  int5 ep={e,0,0,0,0};int count=s_i32_ld_g(gen_addr(ep,counts));int bad=count<1||count>3;
  if(begin[0]==0)s_i32_st_tnsr_s(ep,status,bad);
  for(int m=0;m<3;++m){
   int5 rp={m,e,0,0,0};int row=s_i32_ld_g(gen_addr(rp,row_ids));bool valid=m<count&&!bad&&row>=0&&row<T;
   for(int v=begin[0];v<end[0];++v){
    int5 p={v*128,row,0,0,0},q={v*128,flat?e*3+m:m,flat?0:e,0,0};
    bfloat128 income=(bfloat128)(ushort128)(flat?0:0x7fc1);
    bfloat128 x=v_bf16_ld_tnsr_b(p,tokens,0,income,valid);
    v_bf16_st_tnsr(q,output,x);
   }
  }
 }
}
