// Experimental route-weight preload only. Keep the original slot FMA order,
// N128 tasks, and even/odd FP32 partial-to-natural-lane restoration.
#define GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
void main(tensor partial,tensor routing,tensor directions,tensor output,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int h=get_dim_size(output,0);
 int5 z={0,0,0,0,0};uchar256 de=v_u8_ld_tnsr_b(z,directions);z[1]=1;uchar256 do_=v_u8_ld_tnsr_b(z,directions);
 for(int task=begin[0];task<end[0];++task){
  int token=task/(h/128),block=task%(h/128);
  int5 r={0,token,0,0,0};
  float64 weights=v_f32_ld_tnsr_partial_b(r,routing,7,0);
  // Replicate the first dual group, then its lower eight lanes. No arithmetic
  // or conversion touches routing values; SHUFFLE selects their original bits.
  weights=v_f32_mov_dual_group_all_b(weights,-1,0,0,0,0,GROUPS,0);
  weights=v_f32_mov_group_b(weights,-1,SW_GROUP1_EN|SW_DUAL_GROUP0_EN|SW_DUAL_GROUP1_EN|SW_DUAL_GROUP2_EN|SW_DUAL_GROUP3_EN,weights);
  float64 even=0,odd=0;
#ifdef GK_COMBINE_UNROLL8
  #pragma loop_unroll(8)
  for(int slot=0;slot<8;++slot){
#else
  for(int slot=0;slot<experts;++slot){
#endif
   int5 p={(token*experts+slot)*h+block*128,0,0,0,0};
   uint64 control=(slot|0x80)*0x01010101u;
   float64 weight=v_f32_shuffle_b(weights,(uchar256)control,0,0);
   even=v_f32_mac_b(v_f32_ld_tnsr_b(p,partial),weight,even);p[0]+=64;
   odd=v_f32_mac_b(v_f32_ld_tnsr_b(p,partial),weight,odd);
  }
  float64 firste=v_f32_mov_dual_group_all_b(even,-1,0,0,1,1,GROUPS,0),firsto=v_f32_mov_dual_group_all_b(odd,-1,0,0,1,1,GROUPS,0);
  float64 first=v_f32_shuffle_b(firste,de,0,0);
  first+=v_f32_shuffle_b(firsto,do_,0,0);
  float64 seconde=v_f32_mov_dual_group_all_b(even,-1,2,2,3,3,GROUPS,0),secondo=v_f32_mov_dual_group_all_b(odd,-1,2,2,3,3,GROUPS,0);
  float64 second=v_f32_shuffle_b(seconde,de,0,0);
  second+=v_f32_shuffle_b(secondo,do_,0,0);
  int5 dst={block*128,token,0,0,0};v_f32_st_tnsr(dst,output,first);
  dst[0]+=64;v_f32_st_tnsr(dst,output,second);
 }
}
