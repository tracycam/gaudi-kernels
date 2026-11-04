// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"
#include "../moe_router/precise_ratio.h"

// Match SiLU's BF16 result before the BF16 gate/up product.
void main(tensor gate_up, tensor output) {
  const int width = get_dim_size(output, 0);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int block = begin[0]; block < end[0]; ++block) {
      int5 gate_pos = {block * 128, row, 0, 0, 0};
      int5 up_pos = {block * 128 + width, row, 0, 0, 0};
      int count = s_i32_min(width - gate_pos[0], 128) - 1;
      float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(gate_pos, gate_up, (char)count, 0));
      float128 up = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(up_pos, gate_up, (char)count, 0));
      float128 silu = {router_precise_ratio(gate.v1, 1.f + tpc_exp_f32_safe(-gate.v1)),
                       router_precise_ratio(gate.v2, 1.f + tpc_exp_f32_safe(-gate.v2))};
      bfloat128 rounded = convert_float128_to_bfloat128(silu, SW_RHNE);
      float128 activated = v_convert_bf16_to_f32_all_b(rounded);
      float128 product = {activated.v1 * up.v1, activated.v2 * up.v2};
      v_bf16_st_tnsr_partial(gate_pos, output, convert_float128_to_bfloat128(product, SW_RHNE), (char)count, 0);
    }
  }
}
