// SPDX-License-Identifier: Apache-2.0
// Decode prepared 128x128 block-FP8 weights into 2K FP8_143 cross residual components
// ([256 * dw / s_blk, w0 / s_blk] at group * 256 + {0, 128}).
#include "activation_quant_common.h"

void main(tensor input, tensor scales, tensor blk_scale, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int group = begin[0]; group < end[0]; ++group) {
    int base = group * 256;
    for (int block = begin[1]; block < end[1]; ++block) {
      int5 scale_pos = {group, block, 0, 0, 0};
      float scale = s_f32_ld_g(gen_addr(scale_pos, scales));
      int5 bp = {0, block, 0, 0, 0};
      float64 bscale = v_f32_ld_g(gen_addr(bp, blk_scale));
      float64 inv_bscale = v_reciprocal_f32(bscale);
      inv_bscale = v_f32_mac_b(v_f32_mac_b(-bscale, inv_bscale, 1.0f), inv_bscale, inv_bscale);
      float64 inv_bscale256 = inv_bscale * 256.0f;

      #pragma clang loop unroll_count(4)
      for (int i = 0; i < 128; ++i) {
        int row = block * 128 + i;
        int5 source = {0, row, group, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(source, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
        float128 wide = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(packed));
        wide.v1 *= scale;
        wide.v2 *= scale;
        bfloat128 high = convert_float128_to_bfloat128(wide, SW_RHNE);
        float128 high_wide = v_convert_bf16_to_f32_all_b(high);
        float128 bdw = {(wide.v1 - high_wide.v1) * inv_bscale256, (wide.v2 - high_wide.v2) * inv_bscale256};
        float128 bw0 = {high_wide.v1 * inv_bscale, high_wide.v2 * inv_bscale};

        int5 dst = {base, row, 0, 0, 0};
        v_f8_st_tnsr(dst, output, fp8_143_pack256_all_clamped(bdw, bw0));
      }
    }
  }
}
