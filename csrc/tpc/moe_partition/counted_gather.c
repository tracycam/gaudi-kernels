// Checked row count precedes the sparse-map read. Mode 0 writes invalid rows
// as zero, mode 1 skips them, mode 2 writes NaN to test the gate mask.
void main(tensor input,tensor row_map,tensor valid_rows,tensor status,tensor output,
          int routes,int capacity,int slot_begin,int padding_mode){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status)),tokens=get_dim_size(input,1),total=tokens*routes;
 bfloat128 fill=padding_mode==2?(bfloat128)(ushort128)0x7fc0:(bfloat128)0;
 for(int local=b[2];local<end[2];++local){
  int5 ep={slot_begin+local,0,0,0,0};int valid=s_i32_ld_g(gen_addr(ep,valid_rows));
  int row_end=(!bad&&padding_mode==1&&valid<end[0])?valid:end[0];
  for(int row=b[0];row<row_end;++row){
   int route=-1;
   if(!bad&&row<valid){int5 rp={(slot_begin+local)*capacity+row,0,0,0,0};route=s_i32_ld_g(gen_addr(rp,row_map));}
   if(route>=0&&route<total){
    int token=routes==8?route>>3:routes==4?route>>2:routes==2?route>>1:routes==1?route:route/routes;
    for(int block=b[1];block<end[1];++block){
     int5 src={block*512,token,0,0,0},dst={block*512,row,local,0,0};
     bfloat128 v0=v_bf16_ld_tnsr_b(src,input);src[0]+=128;
     bfloat128 v1=v_bf16_ld_tnsr_b(src,input);src[0]+=128;
     bfloat128 v2=v_bf16_ld_tnsr_b(src,input);src[0]+=128;
     bfloat128 v3=v_bf16_ld_tnsr_b(src,input);
     v_bf16_st_tnsr(dst,output,v0);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,v1);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,v2);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,v3);
    }
   }else if(padding_mode!=1){
    for(int block=b[1];block<end[1];++block){
     int5 dst={block*512,row,local,0,0};
     v_bf16_st_tnsr(dst,output,fill);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,fill);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,fill);dst[0]+=128;
     v_bf16_st_tnsr(dst,output,fill);
    }
   }
  }
 }
}

