// SPDX-License-Identifier: Apache-2.0
void main(tensor x,tensor weight,tensor scales,tensor bias,tensor output){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int groups=get_dim_size(weight,2),outputs=get_dim_size(output,0);
 uint64 lane=read_lane_id_4b_b();
 for(int chunk=begin[0];chunk<end[0];++chunk){
  float64 out0=0,out1=0;
  for(int n=0;n<128;++n){
   int row=chunk*128+n;
   float64 acc=0;
   for(int g=0;g<groups;++g){
    int5 sp={g,row/128,0,0,0};float scale=s_f32_ld_g(gen_addr(sp,scales));
    int5 wp={0,row,g,0,0};
    minifloat256 packed=v_f8_ld_tnsr_b(wp,weight,SW_UNPACK|SW_UNPCK_8_TO_16);
    bfloat128 decoded=v_convert_f8_to_bf16_b(packed);
    float128 w=convert_bfloat128_to_float128(decoded,SW_LINEAR);
    int5 xp={g*128,0,0,0,0};float64 x0=v_f32_ld_tnsr_b(xp,x);
    xp[0]+=64;float64 x1=v_f32_ld_tnsr_b(xp,x);
    acc=v_f32_mac_b(w.v1*scale,x0,acc);acc=v_f32_mac_b(w.v2*scale,x1,acc);
   }
   int5 bp={row,0,0,0,0};float64 value=v_f32_reduce_add(acc)+s_f32_ld_g(gen_addr(bp,bias));
   if(n<64)out0=v_f32_sel_eq_u32_b(lane,n,value,out0);else out1=v_f32_sel_eq_u32_b(lane,n-64,value,out1);
  }
  int5 dst={chunk*128,0,0,0,0};v_f32_st_tnsr(dst,output,out0);
  dst[0]+=64;v_f32_st_tnsr(dst,output,out1);
 }
}
