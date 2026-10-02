// Ascending original route-slot FP32 FMA, including changed routes on replay.
void main(tensor partial,tensor routing,tensor inverse,tensor status,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status)),R=get_dim_size(routing,0),rows=get_dim_size(partial,1);
 for(int t=b[1];t<end[1];++t)for(int block=b[0];block<end[0];++block){
  float64 sum=0;
  if(bad)sum=(float64)(uint64)0x7fc00000;
  else for(int r=0;r<R;++r){
   int5 ip={t*R+r,0,0,0,0},rp={r,t,0,0,0};int row=s_i32_ld_g(gen_addr(ip,inverse));
   if(row<0||row>=rows){sum=(float64)(uint64)0x7fc00000;break;}
   int5 src={block*64,row,0,0,0};float weight=s_f32_ld_g(gen_addr(rp,routing));
   sum=v_f32_mac_b(v_f32_ld_tnsr_b(src,partial),weight,sum);
  }
  int5 dst={block*64,t,0,0,0};v_f32_st_tnsr(dst,output,sum);
 }
}
