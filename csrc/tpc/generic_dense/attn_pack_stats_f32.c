// SPDX-License-Identifier: Apache-2.0
// Pack row-broadcast SRAM statistics into contiguous 64-row vectors.
void main(tensor input_max, tensor input_sum, tensor output_max, tensor output_sum) {
  int5 begin=get_index_space_offset(), end=begin+get_index_space_size();
  int rows=get_dim_size(input_max,1);
  uint64 lanes=read_lane_id_4b_b();
  for (int head=begin[0]; head<end[0]; ++head) {
    for (int block=begin[1]; block<end[1]; ++block) {
      float64 m=-1.0f/0.0f, l=0.f;
      int count=s_i32_min(64,rows-block*64);
      for (int item=0; item<count; ++item) {
        int5 p={0,block*64+item,head,0,0};
        float64 a=v_f32_ld_tnsr_b(p,input_max), b=v_f32_ld_tnsr_b(p,input_sum);
        m=v_f32_sel_eq_u32_b(lanes,item,a,m); l=v_f32_sel_eq_u32_b(lanes,item,b,l);
      }
      int5 out={block*64,head,0,0,0};
      v_f32_st_tnsr(out,output_max,m); v_f32_st_tnsr(out,output_sum,l);
    }
  }
}
