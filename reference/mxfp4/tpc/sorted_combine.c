// FP32 MAC accumulators store even lanes then odd lanes within each 128 rows.
// Restore natural row order without a BF16 rounding step before all-reduce.
void main(tensor partial,tensor routing,tensor directions,tensor inverse,tensor output,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int h=get_dim_size(output,0);
 const int flags=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
 int5 z={0,0,0,0,0};uchar256 de=v_u8_ld_tnsr_b(z,directions);z[1]=1;uchar256 do_=v_u8_ld_tnsr_b(z,directions);
 for(int task=begin[0];task<end[0];++task){
  int token=task/(h/128),block=task%(h/128);
  float64 even=0,odd=0;
  for(int slot=0;slot<experts;++slot){
   int5 w={slot,token,0,0,0};int mapped=s_i32_ld_g(gen_addr(w,inverse));
   int5 p={mapped*h+block*128,0,0,0,0};
   float weight=(float)s_bf16_ld_g(gen_addr(w,routing));
   even=v_f32_mac_b(v_f32_ld_tnsr_b(p,partial),weight,even);p[0]+=64;
   odd=v_f32_mac_b(v_f32_ld_tnsr_b(p,partial),weight,odd);
  }
  float64 firste=v_f32_mov_dual_group_all_b(even,0xffffffff,0,0,1,1,flags,(float64)0), firsto=v_f32_mov_dual_group_all_b(odd,0xffffffff,0,0,1,1,flags,(float64)0);
  float64 first=v_f32_shuffle_b(firste,de,0,(float64)0);
  first+=v_f32_shuffle_b(firsto,do_,0,(float64)0);
  float64 seconde=v_f32_mov_dual_group_all_b(even,0xffffffff,2,2,3,3,flags,(float64)0), secondo=v_f32_mov_dual_group_all_b(odd,0xffffffff,2,2,3,3,flags,(float64)0);
  float64 second=v_f32_shuffle_b(seconde,de,0,(float64)0);
  second+=v_f32_shuffle_b(secondo,do_,0,(float64)0);
  int5 dst={block*128,token,0,0,0};v_f32_st_tnsr(dst,output,first);
  dst[0]+=64;v_f32_st_tnsr(dst,output,second);
 }
}
