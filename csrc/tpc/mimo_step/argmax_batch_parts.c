// SPDX-License-Identifier: Apache-2.0
// Reduce rank-major vocabulary shards independently for each request row.
void main(tensor logits, tensor partial_max, tensor partial_id, tensor partial_bad) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 uint64 lane=read_lane_id_4b_b();
 for(int row=begin[1];row<end[1];++row) for(int part=begin[0];part<end[0];++part){
  float64 best=-1.0f/0.0f;
  int64 ids=2147483647, bad=0;
  int first=part*50, last=s_i32_min(first+50,1192);
  for(int block=first;block<last;++block){
   int rank=block/149, local=block%149;
   int5 p={local*128,row,rank,0,0};
   float128 x=convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p,logits),SW_LINEAR);
   int64 candidate=(int64)lane+block*128;
   ids=v_i32_sel_grt_f32_b(x.v1,best,candidate,ids);
   best=v_f32_max_b(best,x.v1);
   bad=v_i32_max_b(bad,v_i32_sel_eq_u32_b((uint64)x.v1&0x7f800000,0x7f800000,1,0));
   candidate+=64;
   ids=v_i32_sel_grt_f32_b(x.v2,best,candidate,ids);
   best=v_f32_max_b(best,x.v2);
   bad=v_i32_max_b(bad,v_i32_sel_eq_u32_b((uint64)x.v2&0x7f800000,0x7f800000,1,0));
  }
  float64 maximum=v_f32_reduce_max(best);
  ids=v_i32_sel_eq_f32_b(best,maximum,ids,2147483647);
  ids=v_i32_reduce_min(ids);
  bad=v_i32_reduce_max(bad);
  int5 out={part,row,0,0,0};
  v_f32_st_tnsr_partial(out,partial_max,maximum,0,0);
  v_i32_st_tnsr_partial(out,partial_id,ids,0,0);
  v_i32_st_tnsr_partial(out,partial_bad,bad,0,0);
 }
}
