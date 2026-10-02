// Explicit post-vendor-TopK candidate: FP32 original scores, ordered I32 IDs.
// Six contiguous score vectors replace dependent scalar ID->score loads.
#include "precise_ratio.h"
#define ALL_GROUPS (SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3)
void main(tensor scores,tensor ids,tensor weights,tensor ids_out,tensor sum_out,
          float factor,int renormalize,int apply_scale){
 int5 zero={0,0,0,0,0};uint64 lane=read_lane_id_4b_b();
 int64 indices=v_i32_ld_tnsr_partial_b(zero,ids,7,0);
 float64 id_bits=v_f32_mov_dual_group_all_b((float64)indices,-1,0,0,0,0,ALL_GROUPS,0);
 uint64 copied=(uint64)id_bits;
 // F32 SHUFFLE uses element indices and a source-group bit, not byte offsets.
 uint64 control=(copied&7)|0x80u;
 control=v_u32_or_b(control,v_u32_shl_b(control,8));
 control=v_u32_or_b(control,v_u32_shl_b(control,16));
 uint64 wanted_group=copied>>4,local_group=(lane>>4)&3;
 uint64 gathered=0;
 #pragma loop_unroll(6)
 for(int block=0;block<6;++block){
  int5 p={block*64,0,0,0,0};float64 source=v_f32_ld_tnsr_b(p,scores);
  // Replicate each source half across both groups before heterogeneous
  // selection. Direct heterogeneous source-group controls failed the actual
  // ISA gate; half replication plus integer selection keeps every slot exact.
  float64 lower=v_f32_mov_group_b(source,-1,SW_GROUP1_EN|SW_DUAL_GROUP0_EN|SW_DUAL_GROUP1_EN|SW_DUAL_GROUP2_EN|SW_DUAL_GROUP3_EN,source);
  float64 upper=v_f32_mov_group_b(source,-1,SW_GROUP0_EN|SW_DUAL_GROUP0_EN|SW_DUAL_GROUP1_EN|SW_DUAL_GROUP2_EN|SW_DUAL_GROUP3_EN,source);
  uint64 low=v_u32_shuffle_b((uint64)lower,(uchar256)control,0,0);
  uint64 high=v_u32_shuffle_b((uint64)upper,(uchar256)control,0,0);
  uint64 shuffled=v_u32_sel_eq_u32_b(copied&8,0,low,high);
  gathered|=v_u32_sel_eq_u32_b(wanted_group,local_group+block*4,shuffled,0);
 }
 float64 swapped=v_f32_mov_dual_group_all_b((float64)gathered,-1,1,0,3,2,ALL_GROUPS,0);
 gathered|=(uint64)swapped;
 swapped=v_f32_mov_dual_group_all_b((float64)gathered,-1,2,3,0,1,ALL_GROUPS,0);
 gathered|=(uint64)swapped;
 float64 values=(float64)v_u32_sel_less_u32_b(lane,8,gathered,0);
 float64 total=v_f32_reduce_add(values);v_f32_st_tnsr_partial(zero,sum_out,total,0,0);
 if(renormalize){
  values=router_precise_ratio(values,total);
 }
 // Preserve the Python factor==1 skip. A C if was if-converted to MUL by 1,
 // which flushes tiny values/sign-zero even when scaling is disabled.
 values=v_f32_mul_b(values,factor,0,values,apply_scale!=0);
 v_f32_st_tnsr_partial(zero,weights,values,7,0);v_i32_st_tnsr_partial(zero,ids_out,indices,7,0);
}
