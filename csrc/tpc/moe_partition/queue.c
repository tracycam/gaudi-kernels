// Stable device compaction, one core. Masked routes remain in the tail so every
// original output is written. Consumers keep their original route order.
void main(tensor ids,tensor gp_order,tensor down_order){
 int T=get_dim_size(ids,1),R=get_dim_size(ids,0),cursor=0;
 int64 lanes=read_lane_id_4b();
 for(int phase=0;phase<2;++phase)for(int t=0;t<T;++t)for(int r=0;r<R;++r){
  int5 p={r,t,0,0,0};int id=s_i32_ld_g(gen_addr(p,ids));
  if((id>=0)==(phase==0)){
   int route=t*R+r;int5 gp={cursor*3,0,0,0,0},down={cursor*12,0,0,0,0};
   // Overlapping stores belong to this one task and proceed in order. Later
   // stores overwrite each tail; tensor bounds suppress the final excess.
   v_i32_st_tnsr(gp,gp_order,lanes+route*3);
   v_i32_st_tnsr(down,down_order,lanes+route*12);++cursor;
  }
 }
}
