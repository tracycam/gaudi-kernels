// Direct route-row gather; no expert scan, CPU route values, or output alias.
void main(tensor input,tensor row_map,tensor status,tensor output,
          int routes,int capacity,int slot_begin){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
#if defined(GK_ROUTE_GATHER_ROW_FIRST) && GK_ROUTE_GATHER_ROW_FIRST
 const int row_dim=0,block_dim=1;
#else
 const int row_dim=1,block_dim=0;
#endif
 int bad=s_i32_ld_g(gen_addr(z,status)),tokens=get_dim_size(input,1);
 for(int local=b[2];local<end[2];++local)for(int row=b[row_dim];row<end[row_dim];++row){
  int5 rp={(slot_begin+local)*capacity+row,0,0,0,0};
  int route=s_i32_ld_g(gen_addr(rp,row_map));
#if defined(GK_ROUTE_GATHER_HOIST) && GK_ROUTE_GATHER_HOIST
  const int valid=!bad&&route>=0&&route<tokens*routes;
  // Positive legal route; top1/2/4/8 need no general integer division.
  int token=0;if(valid)token=routes==8?route>>3:routes==4?route>>2:routes==2?route>>1:routes==1?route:route/routes;
#endif
  for(int block=b[block_dim];block<end[block_dim];++block){
   int5 dst={block*128,row,local,0,0};bfloat128 value=0;
#if defined(GK_ROUTE_GATHER_HOIST) && GK_ROUTE_GATHER_HOIST
   if(valid){int5 src={block*128,token,0,0,0};value=v_bf16_ld_tnsr_b(src,input);}
#else
   if(!bad&&route>=0&&route<tokens*routes){int5 src={block*128,route/routes,0,0,0};value=v_bf16_ld_tnsr_b(src,input);}
#endif
   v_bf16_st_tnsr(dst,output,value);
  }
 }
}
