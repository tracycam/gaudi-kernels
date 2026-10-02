// One vector of <=64 actual top8 routes. No 384-expert prefix scan, no atomics.
// Keep reductions in VPU registers; materialized metadata is the VPU->SPU edge.
// Slot = first original route index for that expert. Holes have count=0/map=-1.
void main(tensor ids,tensor experts,tensor counts,tensor rows,tensor inverse,int cap){
 int length=get_dim_size(ids,0),tokens=length/8;int5 z={0,0,0,0,0};
 int64 all=v_i32_ld_tnsr_b(z,ids);uint64 lane=read_lane_id_4b_b();
 float64 index=convert_uint64_to_float64(lane,0);
 int5 b=get_index_space_offset(),end=b+get_index_space_size();
 for(int route=b[0];route<end[0];++route){
  int5 p={route,0,0,0,0};int e=s_i32_ld_g(gen_addr(p,ids));
  float64 candidates=v_f32_sel_eq_i32_b(all,e,index,1024.f);
  candidates=v_f32_sel_less_u32_b(lane,(unsigned)length,candidates,1024.f);
  float64 first=v_f32_reduce_min(candidates);
  float64 hits=v_f32_sel_less_f32_b(candidates,1024.f,1.f,0.f);
  float64 n=v_f32_reduce_add(hits);
  float64 before=v_f32_reduce_add(v_f32_sel_less_u32_b(lane,(unsigned)route,hits,0.f));
  float64 count=v_f32_sel_eq_f32_b(first,(float)route,n,0.f);
  float64 expert=v_f32_sel_eq_f32_b(first,(float)route,(float)e,-1.f);
  float64 inv=first*(float)cap+before;
#ifdef GK_SMALLM_STRIPE
  // Spread first-token leaders over all token-sized ranges. The metadata task
  // and row-map values still name original routes; only expert slots change.
  int64 fi=convert_float64_to_int64(first,SW_RHNE);
  float64 permuted=convert_int64_to_float64(fi&7,0)*(float)tokens+convert_int64_to_float64(fi>>3,0);
  inv=permuted*(float)cap+before;
  int slot=(route&7)*tokens+(route>>3);
#else
  int slot=route;
#endif
  count=v_f32_sel_leq_f32_b(n,(float)cap,count,0.f);
  expert=v_f32_sel_leq_f32_b(n,(float)cap,expert,-1.f);
  inv=v_f32_sel_leq_f32_b(n,(float)cap,inv,-1.f);
  if(e<0||e>=384){count=0;expert=-1;inv=-1;}
  int5 owner={slot,0,0,0,0};
  v_i32_st_tnsr_partial(owner,counts,convert_float64_to_int64(count,SW_RHNE),0,0);
  v_i32_st_tnsr_partial(owner,experts,convert_float64_to_int64(expert,SW_RHNE),0,0);
  v_i32_st_tnsr_partial(p,inverse,convert_float64_to_int64(inv,SW_RHNE),0,0);
  float64 previous=-1;
  for(int r=0;r<cap;++r){
   float64 next=v_f32_reduce_min(v_f32_sel_grt_f32_b(candidates,previous,candidates,1024.f));
   float64 value=v_f32_sel_less_f32_b(next,(float)length,next,-1.f);
   value=v_f32_sel_eq_f32_b(first,(float)route,value,-1.f);
   value=v_f32_sel_leq_f32_b(n,(float)cap,value,-1.f);
   if(e<0||e>=384)value=-1;
   int5 dst={r,slot,0,0,0};
   v_i32_st_tnsr_partial(dst,rows,convert_float64_to_int64(value,SW_RHNE),0,0);
   previous=next;
  }
 }
}
