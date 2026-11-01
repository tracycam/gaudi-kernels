// Keep gate accumulation in FP32 through SiLU and guard Gaudi2 exp boundaries.
#include "../common/tpc_math_safe.h"
void main(tensor partial,tensor valid_rows,tensor status,tensor output,int slot_begin){
 int5 b=get_index_space_offset(),end=b+get_index_space_size(),z={0,0,0,0,0};
 int bad=s_i32_ld_g(gen_addr(z,status)),width=get_dim_size(output,0);
 for(int local=b[2];local<end[2];++local){
  int5 vp={slot_begin+local,0,0,0,0};int valid=bad?0:s_i32_ld_g(gen_addr(vp,valid_rows));
  for(int row=b[1];row<end[1];++row){
   int5 dst0={0,row,local,0,0},dst1={128,row,local,0,0};
   if(row>=valid){
    v_bf16_st_tnsr(dst0,output,(bfloat128)0);
    v_bf16_st_tnsr(dst1,output,(bfloat128)0);
    continue;
   }
   for(int block=0;block<2;++block){
    int5 dst={block*128,row,local,0,0},gp={block*128,row,local,0,0},up={block*128+width,row,local,0,0};
    float128 gs={0},us={0};
    gs.v1+=v_f32_ld_tnsr_b(gp,partial);gp[0]+=64;gs.v2+=v_f32_ld_tnsr_b(gp,partial);
    us.v1+=v_f32_ld_tnsr_b(up,partial);up[0]+=64;us.v2+=v_f32_ld_tnsr_b(up,partial);
    bfloat128 u=v_convert_f32_to_bf16_all_b(us);
    float128 g=gs;
    g.v1=g.v1*v_reciprocal_f32(1.f+tpc_exp_f32_safe(-g.v1));g.v2=g.v2*v_reciprocal_f32(1.f+tpc_exp_f32_safe(-g.v2));
    bfloat128 silu=v_convert_f32_to_bf16_all_b(g);
    uint128 lanes=v_convert_u16_to_u32_all_b((ushort128)(silu*u));
    bfloat128 result=(bfloat128)convert_uint128_to_ushort128(lanes,SW_LINEAR);
    v_bf16_st_tnsr(dst,output,result);
   }
  }
 }
}
