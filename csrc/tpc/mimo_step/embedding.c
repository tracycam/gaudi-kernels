// SPDX-License-Identifier: Apache-2.0
// TP8 M=1 or M=32 embedding: each device token selects a BF16 table row.
// Invalid token IDs set status to one without accessing the table.
void main(tensor table, tensor token, tensor output, tensor status) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int task=begin[0];task<end[0];++task){
  int row=task/48,block=task%48;
  int5 token_pos={row,0,0,0,0};
  int id=s_i32_ld_g(gen_addr(token_pos,token));
  bool valid=id>=0 && id<get_dim_size(table,1);
  int5 dst={block*128,row,0,0,0};
  bfloat128 value=0;
  if(valid){int5 src={block*128,id,0,0,0};value=v_bf16_ld_tnsr_b(src,table);}
  v_bf16_st_tnsr(dst,output,value);
  if(block==0)s_i32_st_g(gen_addr(token_pos,status),valid?0:1);
 }
}
