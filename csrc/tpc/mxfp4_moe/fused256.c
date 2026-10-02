// N256 ownership keeps each nibble-sharing pair together in the original v2
// byte stream. Original BF16 A and FP32 route weights; FP32 final stores only.
void main(tensor packed,tensor scales,tensor activation,tensor lut,tensor ids,
 tensor routing,tensor output,int nblocks,unsigned reciprocal,int experts,unsigned route_reciprocal){
 set_lut_256(lut);int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(activation,0),R=get_dim_size(ids,0);
 for(int task=begin[0];task<end[0];++task){
  int token=task/nblocks,nb=task%nblocks;float128 combined0={0},combined1={0};
  for(int slot=0;slot<R;++slot){int5 rp={slot,token,0,0,0};int e=s_i32_ld_g(gen_addr(rp,ids));if(e<0||e>=experts)continue;
   int mapped=e*nblocks+nb,base=(mapped/2)*K*2+mapped%2,sbase=(mapped/2)*(K/32)*2+mapped%2;
   float weight=s_f32_ld_g(gen_addr(rp,routing));float128 total0={0},total1={0};
   for(int group=0;group<K/32;++group){
    float128 part0={0},part1={0};
    for(int z=0;z<32;++z){int k=group*32+z;int5 wp={0,base+k*2,0,0,0},ap={k,token*R+slot,0,0,0};
     uchar256 raw=v_u8_ld_tnsr_b(wp,packed);ushort256 index=convert_uchar256_to_ushort256(raw,SW_LINEAR);
     bfloat256 q=v_bf16_lookup_2c(index.v1,0,SW_LUT_PTR,(bfloat256){0});bf16 x=s_bf16_ld_g(gen_addr(ap,activation));
     part0=v_bf16_mac_acc32_b(q.v1,x,part0);part1=v_bf16_mac_acc32_b(q.v2,x,part1);
    }
    int5 sp={0,sbase+group*2,0,0,0};ushort256 bits=convert_uchar256_to_ushort256(v_u8_ld_tnsr_b(sp,scales),SW_LINEAR);
    float128 f0=v_convert_bf16_to_f32_all_b((bfloat128)(bits.v1<<7)),f1=v_convert_bf16_to_f32_all_b((bfloat128)(bits.v2<<7));
    total0.v1=v_f32_mac_b(part0.v1,f0.v1,total0.v1);total0.v2=v_f32_mac_b(part0.v2,f0.v2,total0.v2);
    total1.v1=v_f32_mac_b(part1.v1,f1.v1,total1.v1);total1.v2=v_f32_mac_b(part1.v2,f1.v2,total1.v2);
   }
   combined0.v1=v_f32_mac_b(total0.v1,weight,combined0.v1);combined0.v2=v_f32_mac_b(total0.v2,weight,combined0.v2);
   combined1.v1=v_f32_mac_b(total1.v1,weight,combined1.v1);combined1.v2=v_f32_mac_b(total1.v2,weight,combined1.v2);
  }
  int5 dst={nb*256,token,0,0,0};v_f32_st_tnsr(dst,output,combined0.v1);dst[0]+=64;v_f32_st_tnsr(dst,output,combined0.v2);dst[0]+=64;v_f32_st_tnsr(dst,output,combined1.v1);dst[0]+=64;v_f32_st_tnsr(dst,output,combined1.v2);
 }
}
