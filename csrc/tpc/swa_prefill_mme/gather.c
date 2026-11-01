// SPDX-License-Identifier: Apache-2.0
// Gather the union of the first and last query's 128-token windows. Four
// batches of 128 queries each need at most three physical pages, including
// chunks starting at a 32-token (rather than 128-token) boundary.
void main(tensor key, tensor value, tensor pages, tensor groups, tensor positions,
          tensor gathered_key, tensor gathered_value) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int slots=get_dim_size(key,2);
 for(int batch=begin[1];batch<end[1];++batch){
  int row=batch*128,last=row+127;
  int5 p={row,0,0,0,0};int first_pos=s_i32_ld_g(gen_addr(p,positions));
  p[0]=last;int last_pos=s_i32_ld_g(gen_addr(p,positions));
  int first_page=s_i32_max(0,first_pos-127)>>7;
  int last_page=s_i32_max(0,last_pos-127)>>7;
  for(int token=begin[0];token<end[0];++token){
   int logical=first_page+token/128,physical=-1;
   for(int side=0;side<2;++side){int r=side?last:row,base=side?last_page:first_page;
    for(int j=0;j<2;++j){int5 m={2*r+j,0,0,0,0};
     int group=s_i32_ld_g(gen_addr(m,groups));
     if(logical==base+j&&group==r)physical=s_i32_ld_g(gen_addr(m,pages));
    }
   }
   bfloat128 k0=0,k1=0,v=0;
   if(physical>=0&&physical<slots/128){int5 src={0,0,physical*128+(token%128),0,0};
    k0=v_bf16_ld_tnsr_b(src,key);src[0]=128;k1=v_bf16_ld_tnsr_partial_b(src,key,63,0);
    src[0]=0;v=v_bf16_ld_tnsr_b(src,value);
   }
   int5 dst={0,token,batch,0,0};v_bf16_st_tnsr(dst,gathered_key,k0);
   dst[0]=128;v_bf16_st_tnsr_partial(dst,gathered_key,k1,63,0);
   dst[0]=0;v_bf16_st_tnsr(dst,gathered_value,v);
  }
 }
}
