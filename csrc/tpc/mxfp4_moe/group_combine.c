// Restore original route order after expert-batched MME. The returned device
// overflow tensor is mandatory; an overflow route makes its output token NaN.
void main(tensor grouped,tensor ids,tensor routing,tensor inverse,tensor overflow,
          tensor output,int capacity,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int R=get_dim_size(ids,0);
 for(int token=begin[1];token<end[1];++token)for(int block=begin[0];block<end[0];++block){float64 sum=0;
  for(int slot=0;slot<R;++slot){int5 rp={slot,token,0,0,0};int e=s_i32_ld_g(gen_addr(rp,ids));if(e<0||e>=experts)continue;
   int5 ep={e,0,0,0,0};int bad=s_i32_ld_g(gen_addr(ep,overflow)),row=s_i32_ld_g(gen_addr(rp,inverse));
   if(bad||row<0||row>=capacity){sum=(float64)(uint64)0x7fc00000;break;}
   int5 src={block*64,row,e,0,0};float weight=s_f32_ld_g(gen_addr(rp,routing));sum=v_f32_mac_b(v_f32_ld_tnsr_b(src,grouped),weight,sum);
  }
  int5 dst={block*64,token,0,0,0};v_f32_st_tnsr(dst,output,sum);
 }
}
