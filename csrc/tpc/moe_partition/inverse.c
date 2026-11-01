// Route-balanced index space with 64-aligned vector load/store fast path.
void main(tensor ids,tensor flat_ids,tensor prefix,tensor status,tensor chunk_offsets,tensor inverse,int rows,int capacity){
 int length=get_dim_size(flat_ids,0),experts=get_dim_size(prefix,0)-1;
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size(),zero={0,0,0,0,0},ep_end={experts,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(zero,status)),total_sel=bad?0:s_i32_ld_g(gen_addr(ep_end,prefix)),cached_chunk=-1;
 uint64 lane=read_lane_id_4b_b();int64 packed=0;
 for(int q=begin[0];q<end[0];++q){
  if(q>=length)break;
  if(((q&63)==0)&&(q+64<=end[0])&&(q+64<=length)){
   int5 src={q,0,0,0,0};int64 out=-1;
   if(total_sel>0){
    int chunk=q>>6;
    packed=v_i32_ld_tnsr_b(src,flat_ids);cached_chunk=chunk;
    int64 rank=0;
    for(int l=0;l<64;++l){
     int5 dst={q+l,0,0,0,0};int expert=s_i32_ld_g(gen_addr(dst,flat_ids));
     if(expert>=0&&expert<experts){
      int5 p={expert,0,0,0,0},next={expert+1,0,0,0,0};
      int p0=s_i32_ld_g(gen_addr(p,prefix)),p1=s_i32_ld_g(gen_addr(next,prefix));
      if(p1>p0){
       int5 off={chunk,expert,0,0,0};int base=p0*rows+s_i32_ld_g(gen_addr(off,chunk_offsets));
       if(base>=0&&base<capacity*rows){
        out=v_i32_sel_eq_u32_b(lane,(unsigned)l,rank+base,out);
        rank=v_i32_sel_eq_i32_b(packed,expert,rank+1,rank);
       }
      }
     }
    }
   }
   v_i32_st_tnsr(src,inverse,out);
   q+=63;continue;
  }
  int5 dst={q,0,0,0,0};int64 target=-1;
  if(total_sel>0){
   int expert=s_i32_ld_g(gen_addr(dst,flat_ids));
   if(expert>=0&&expert<experts){
    int5 p={expert,0,0,0,0},next={expert+1,0,0,0,0};
    int p0=s_i32_ld_g(gen_addr(p,prefix)),p1=s_i32_ld_g(gen_addr(next,prefix));
    if(p1>p0){
     int chunk=q>>6,first=chunk<<6;
     int5 off={chunk,expert,0,0,0};
     int base=p0*rows+s_i32_ld_g(gen_addr(off,chunk_offsets));
     if(base>=0&&base<capacity*rows){
      if(chunk!=cached_chunk){int5 src={first,0,0,0,0};packed=v_i32_ld_tnsr_b(src,flat_ids);cached_chunk=chunk;}
      float64 hits=v_f32_sel_eq_i32_b(packed,expert,1.f,0.f);
      hits=v_f32_sel_less_u32_b(lane,(unsigned)(q-first),hits,0.f);
      float64 sum=v_f32_reduce_add(hits)+(float)base;
      target=convert_float64_to_int64(sum,SW_RHNE);
     }
    }
   }
  }
  v_i32_st_tnsr_partial(dst,inverse,target,0,0);
 }
}

