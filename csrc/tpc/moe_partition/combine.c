// Four independent vectors share each scalar route/weight load. Each output
// element retains ascending original route-slot FP32 FMA order.
void main(tensor partial,tensor routing,tensor inverse,tensor status,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status)),R=get_dim_size(routing,0),rows=get_dim_size(partial,1);
 for(int t=b[1];t<end[1];++t)for(int block=b[0];block<end[0];++block){
  float64 sum=0,s1=0,s2=0,s3=0;
  if(bad)sum=s1=s2=s3=(float64)(uint64)0x7fc00000;
  else for(int r=0;r<R;++r){
   int5 ip={t*R+r,0,0,0,0},rp={r,t,0,0,0};int row=s_i32_ld_g(gen_addr(ip,inverse));
   // -1 is a validated route owned by another bucket or the TPC path.
   if(row==-1)continue;
   if(row<0||row>=rows){sum=s1=s2=s3=(float64)(uint64)0x7fc00000;break;}
   int5 src={block*256,row,0,0,0};float weight=s_f32_ld_g(gen_addr(rp,routing));
   float64 v0=v_f32_ld_tnsr_b(src,partial);src[0]+=64;
   float64 v1=v_f32_ld_tnsr_b(src,partial);src[0]+=64;
   float64 v2=v_f32_ld_tnsr_b(src,partial);src[0]+=64;
   float64 v3=v_f32_ld_tnsr_b(src,partial);
   sum=v_f32_mac_b(v0,weight,sum);s1=v_f32_mac_b(v1,weight,s1);
   s2=v_f32_mac_b(v2,weight,s2);s3=v_f32_mac_b(v3,weight,s3);
  }
  int5 dst={block*256,t,0,0,0};v_f32_st_tnsr(dst,output,sum);dst[0]+=64;
  v_f32_st_tnsr(dst,output,s1);dst[0]+=64;
  v_f32_st_tnsr(dst,output,s2);dst[0]+=64;v_f32_st_tnsr(dst,output,s3);
 }
}
