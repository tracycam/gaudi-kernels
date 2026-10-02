// SPDX-License-Identifier: Apache-2.0
// MiMo TP8: [16*192 Q, 192 K, 128 V], NeoX rotation of first64/head.
// Produces small contiguous tensors. KV caches are never inputs or outputs.
#ifndef ROPE_F32
#define ROPE_F32 0
#endif
#ifndef VSCALE_F32
#define VSCALE_F32 0
#endif
#ifndef CACHE_INPUT
#define CACHE_INPUT 0
#endif
#define ALL_GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
static inline bfloat128 rotate(bfloat128 x,bfloat128 cosine,bfloat128 sine){
 ushort128 lane=read_lane_id_2b_b();
 // Each dual group contains32 BF16 elements. Swap0/1 to exchange the
 // two32-element NeoX halves; other lanes are later selected from x unchanged.
 bfloat128 partner=v_bf16_mov_dual_group_all_b(x,-1,1,0,3,2,ALL_GROUPS,0);
 ushort128 sign=v_u16_sel_less_u16_b(lane&63,32,0x8000,0);
 partner=(bfloat128)((ushort128)partner^sign);
#if ROPE_F32
 float128 a=v_convert_bf16_to_f32_all_b(x),b=v_convert_bf16_to_f32_all_b(partner);
 float128 c=v_convert_bf16_to_f32_all_b(cosine),s=v_convert_bf16_to_f32_all_b(sine);
 float128 y={a.v1*c.v1+b.v1*s.v1,a.v2*c.v2+b.v2*s.v2};
 bfloat128 rotated=convert_float128_to_bfloat128(y,SW_RHNE);
#else
 bfloat128 ac=x*cosine,bs=partner*sine;
 bfloat128 rotated=ac+bs;
#endif
 return (bfloat128)v_u16_sel_less_u16_b(lane,64,(ushort128)rotated,(ushort128)x);
}
void main(tensor qkv,tensor cosines,tensor sines,tensor query,tensor key,tensor value,float value_scale){
 int5 start=get_index_space_offset(),end=start+get_index_space_size();
 for(int row=start[0];row<end[0];++row){
#if CACHE_INPUT
  // cosines is the actual layer cache [P,64]: cos32 followed by sin32.
  // sines is original I32 position metadata, no gather/expanded cos/sin nodes.
  bool column=get_dim_size(sines,0)==1;int5 pp={column?0:row,column?row:0,0,0,0};
  int pos=s_i32_ld_g(gen_addr(pp,sines));
  if(pos<0||pos>=get_dim_size(cosines,1)){
   // Defined fail-closed output for invalid metadata, with no out-of-bounds read.
   // Qualified production calls require valid positions; no clamping is used.
   for(int i=0;i<3072;i+=128){int5 p={i,row,0,0,0};v_bf16_st_tnsr(p,query,0);}
   int5 p={0,0,row,0,0};v_bf16_st_tnsr(p,key,0);v_bf16_st_tnsr(p,value,0);p[0]=128;v_bf16_st_tnsr_partial(p,key,0,63,0);continue;
  }
  int5 rp={0,pos,0,0,0};bfloat128 cache=v_bf16_ld_tnsr_partial_b(rp,cosines,63,0);
  bfloat128 c=v_bf16_mov_dual_group_all_b(cache,-1,0,0,0,0,ALL_GROUPS,0);
  bfloat128 s=v_bf16_mov_dual_group_all_b(cache,-1,1,1,1,1,ALL_GROUPS,0);
#else
  int5 rp={0,0,row,0,0};bfloat128 c=v_bf16_ld_tnsr_partial_b(rp,cosines,63,0),s=v_bf16_ld_tnsr_partial_b(rp,sines,63,0);
#endif
  for(int head=0;head<17;++head){
   int5 p={head*192,row,0,0,0};bfloat128 x=v_bf16_ld_tnsr_b(p,qkv);bfloat128 y=rotate(x,c,s);
   p[0]+=128;bfloat128 tail=v_bf16_ld_tnsr_partial_b(p,qkv,63,0);
   int5 dst={0,head<16?head:0,row,0,0};
   if(head<16){
#if CACHE_INPUT
    dst[0]=head*192;dst[1]=row;dst[2]=0;v_bf16_st_tnsr(dst,query,y);dst[0]+=128;v_bf16_st_tnsr_partial(dst,query,tail,63,0);
#else
    v_bf16_st_tnsr(dst,query,y);dst[0]=128;v_bf16_st_tnsr_partial(dst,query,tail,63,0);
#endif
   }
   else{v_bf16_st_tnsr(dst,key,y);dst[0]=128;v_bf16_st_tnsr_partial(dst,key,tail,63,0);}
  }
  int5 vp={3264,row,0,0,0};bfloat128 v=v_bf16_ld_tnsr_b(vp,qkv);
#if VSCALE_F32
  float128 f=v_convert_bf16_to_f32_all_b(v);f.v1*=value_scale;f.v2*=value_scale;v=convert_float128_to_bfloat128(f,SW_RHNE);
#else
  v*= (bf16)value_scale;
#endif
  int5 dst={0,0,row,0,0};v_bf16_st_tnsr(dst,value,v);
 }
}
