// One route per task. No vector lane extraction into SPU, no expert scan.
void main(tensor ids,tensor flat_ids,tensor prefix,tensor status,tensor chunk_offsets,tensor inverse,int rows,int capacity){
 int length=get_dim_size(flat_ids,0),experts=get_dim_size(prefix,0)-1;
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size(),zero={0,0,0,0,0};int bad=s_i32_ld_g(gen_addr(zero,status));uint64 lane=read_lane_id_4b_b();
 for(int q=begin[0];q<end[0];++q){
  int5 dst={q,0,0,0,0};int64 target=-1;
  if(!bad){int expert=s_i32_ld_g(gen_addr(dst,flat_ids));
   if(expert>=0&&expert<experts){int chunk=q/64,first=chunk*64;int5 p={expert,0,0,0,0},off={chunk,expert,0,0,0},src={first,0,0,0,0};int base=s_i32_ld_g(gen_addr(p,prefix))*rows+s_i32_ld_g(gen_addr(off,chunk_offsets));
    if(base>=0&&base<capacity*rows){int64 packed=v_i32_ld_tnsr_b(src,flat_ids);float64 hits=v_f32_sel_eq_i32_b(packed,expert,1.f,0.f);hits=v_f32_sel_less_u32_b(lane,(unsigned)(q-first),hits,0.f);hits=v_f32_sel_less_u32_b(lane,(unsigned)(length-first),hits,0.f);float64 sum=v_f32_reduce_add(hits)+(float)base;target=convert_float64_to_int64(sum,SW_RHNE);}
   }
  }
  v_i32_st_tnsr_partial(dst,inverse,target,0,0);
 }
}
