// Checked row count precedes the sparse-map read. Mode 0 writes invalid rows
// as zero, mode 1 skips them, mode 2 writes NaN to test the gate mask.
void main(tensor input,tensor row_map,tensor valid_rows,tensor status,tensor output,
          int routes,int capacity,int slot_begin,int padding_mode){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status)),tokens=get_dim_size(input,1);
 for(int local=b[2];local<end[2];++local){
  int5 ep={slot_begin+local,0,0,0,0};int valid=s_i32_ld_g(gen_addr(ep,valid_rows));
  for(int row=b[0];row<end[0];++row){
   int route=-1;
   if(!bad&&row<valid){int5 rp={(slot_begin+local)*capacity+row,0,0,0,0};route=s_i32_ld_g(gen_addr(rp,row_map));}
   if((route<0||route>=tokens*routes)&&padding_mode==1)continue;
   int token=0;if(route>=0&&route<tokens*routes)token=routes==8?route>>3:routes==4?route>>2:routes==2?route>>1:routes==1?route:route/routes;
   for(int block=b[1];block<end[1];++block){
    bfloat128 value=padding_mode==2?(bfloat128)(ushort128)0x7fc0:(bfloat128)0;
    if(route>=0&&route<tokens*routes){int5 src={block*128,token,0,0,0};value=v_bf16_ld_tnsr_b(src,input);}
    int5 dst={block*128,row,local,0,0};v_bf16_st_tnsr(dst,output,value);
   }
  }
 }
}
