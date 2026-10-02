// Bit-only test of general BF16 activation broadcast from one K32 tensor fetch.
#define ALL_GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
void main(tensor input,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int group=begin[0];group<end[0];++group) {
  int5 in={group*32,0,0,0,0};
  uchar256 values=(uchar256)v_bf16_ld_tnsr_b(in,input);
  values=v_u8_mov_dual_group_all_b(values,-1,0,0,0,0,ALL_GROUPS,0);
  for(int k=0;k<32;++k) {
   uchar256 mask=(uchar256)v_u32_mov_b(0x81808180u+(unsigned)k*0x02020202u);
   bfloat128 value=(bfloat128)v_u8_shuffle_b(values,mask,0,0);
   int5 out={0,group*32+k,0,0,0};v_bf16_st_tnsr(out,output,value);
  }
 }
}
