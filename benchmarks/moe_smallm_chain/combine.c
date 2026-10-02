// Preserve original top8 slot order, FP32 FMA and FP32 output.
void main(tensor partial,tensor routing,tensor inverse,tensor output){
 int5 b=get_index_space_offset(),end=b+get_index_space_size();
#ifndef GK_SMALLM_DENSE
 int cap=get_dim_size(partial,1),slots=get_dim_size(partial,3);
#endif
 for(int t=b[1];t<end[1];++t)for(int block=b[0];block<end[0];++block){
  float64 sum=0;
  for(int r=0;r<8;++r){
   int5 ip={t*8+r,0,0,0,0},rp={r,t,0,0,0};
#ifdef GK_SMALLM_DENSE
   // All allocated rows are written exactly once: valid leader scatter, or
   // that invalid route's own NaN writer. No padded/undefined rows are read.
   int5 src={block*64,0,t*8+r,0,0};
#else
   int index=s_i32_ld_g(gen_addr(ip,inverse));
   if(index<0||index>=slots*cap){sum=(float64)(uint64)0x7fc00000;break;}
   int slot=cap==4?index>>2:index>>3,row=index&(cap-1);
   int5 src={block*64,row,0,slot,0};
#endif
   float weight=s_f32_ld_g(gen_addr(rp,routing));
   sum=v_f32_mac_b(v_f32_ld_tnsr_b(src,partial),weight,sum);
  }
  int5 dst={block*64,t,0,0,0};v_f32_st_tnsr(dst,output,sum);
 }
}
