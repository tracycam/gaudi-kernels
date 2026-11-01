// Stable device compaction, one core. Masked routes remain in the tail so every
// original output is written. Consumers keep their original route order.
void main(tensor ids,tensor gp_order,tensor down_order){
 int T=get_dim_size(ids,1),R=get_dim_size(ids,0),cursor=0;
 for(int phase=0;phase<2;++phase)for(int t=0;t<T;++t)for(int r=0;r<R;++r){
  int5 p={r,t,0,0,0};int id=s_i32_ld_g(gen_addr(p,ids));
  if((id>=0)==(phase==0)){
   int route=t*R+r;
   for(int sub=0;sub<3;++sub){int5 p={cursor*3+sub,0,0,0,0};s_i32_st_g(gen_addr(p,gp_order),route*3+sub);}
   for(int sub=0;sub<12;++sub){int5 p={cursor*12+sub,0,0,0,0};s_i32_st_g(gen_addr(p,down_order),route*12+sub);}
   ++cursor;
  }
 }
}
