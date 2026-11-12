// SPDX-License-Identifier: Apache-2.0
// Emit SRAM-bundled FP16 weight operands from resident 1-byte FP8_143 weights.
#include "activation_quant_common.h"

void main(tensor input, tensor scales, tensor blk_scale, tensor output, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int b0 = begin[1], b_end = end[1];
  for (int group = begin[0]; group < end[0]; ++group) {
    int base = group * 128;
    for (int block = b0; block < b_end; ++block) {
      int row0 = block * 128;
      int5 scale_pos = {group, block, 0, 0, 0};
      float64 s_val = v_f32_ld_g(gen_addr(scale_pos, scales));
      int5 bp = {0, block, 0, 0, 0};
      float64 eff_scale = s_val * fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(bp, blk_scale)));
      #pragma clang loop unroll_count(2)
      for (int i = 0; i < 128; ++i) {
        int row = row0 + i;
        int5 source = {0, row, group, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(source, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
        float128 ws = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(packed));
        ws.v1 *= eff_scale;
        ws.v2 *= eff_scale;
        int5 dst = {base, row, 0, 0, 0};
        v_f16_st_tnsr(dst, output, convert_float128_to_half128(ws, SW_RHNE));
      }
    }
  }
}
