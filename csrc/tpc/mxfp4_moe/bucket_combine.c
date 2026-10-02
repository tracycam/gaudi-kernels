void main(tensor grouped,tensor ids,tensor routing,tensor inverse,tensor status,
          tensor output,int experts){
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();int R=get_dim_size(ids,0),rows=get_dim_size(grouped,1);
 for(int t=begin[1];t<end[1];++t)for(int block=begin[0];block<end[0];++block){float64 sum=0;
  for(int r=0;r<R;++r){int5 rp={r,t,0,0,0};int expert=s_i32_ld_g(gen_addr(rp,ids));if(expert<0||expert>=experts)continue;
   int5 ep={expert,0,0,0,0};if(s_i32_ld_g(gen_addr(ep,status))){sum=(float64)(uint64)0x7fc00000;break;}
   int row=s_i32_ld_g(gen_addr(rp,inverse));if(row<0||row>=rows){sum=(float64)(uint64)0x7fc00000;break;}
   int5 src={block*64,row,0,0,0};float weight=s_f32_ld_g(gen_addr(rp,routing));sum=v_f32_mac_b(v_f32_ld_tnsr_b(src,grouped),weight,sum);
  }
  int5 dst={block*64,t,0,0,0};v_f32_st_tnsr(dst,output,sum);
 }
}
