// Fixed-capacity device gather; expert IDs/counts never leave the graph.
// CAP>=T covers unique-top-k-per-token inputs. Smaller caps return overflow.
void main(tensor input,tensor ids,tensor grouped,tensor inverse,tensor counts,
          tensor overflow,int capacity,int experts,int shared_input){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(input,0),R=get_dim_size(ids,0),T=get_dim_size(ids,1);
 for(int e=begin[0];e<end[0];++e){int count=0;
  for(int token=0;token<T;++token)for(int slot=0;slot<R;++slot){
   int5 route={slot,token,0,0,0};int expert=s_i32_ld_g(gen_addr(route,ids));
   if(expert!=e)continue;
   s_i32_st_g(gen_addr(route,inverse),count);
   if(count<capacity)for(int k=0;k<K;k+=128){int5 src={k,shared_input?token:token*R+slot,0,0,0},dst={k,count,e,0,0};v_bf16_st_tnsr(dst,grouped,v_bf16_ld_tnsr_b(src,input));}
   ++count;
  }
  for(int row=count;row<capacity;++row)for(int k=0;k<K;k+=128){int5 dst={k,row,e,0,0};v_bf16_st_tnsr(dst,grouped,(bfloat128)0);}
  int5 ep={e,0,0,0,0};s_i32_st_g(gen_addr(ep,counts),count);s_i32_st_g(gen_addr(ep,overflow),count>capacity);
 }
}
