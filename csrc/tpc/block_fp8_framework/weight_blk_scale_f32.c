// SPDX-License-Identifier: Apache-2.0
// Per-128-column power-of-2 block scale (>= max_g |scales[g, block]| / 64, or 1.0f if max == 0)
// for SRAM-bundled 1K FP16 block-FP8 prefill.
#include "activation_quant_common.h"

void main(tensor scales, tensor blk_scale, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(scales, 0);
  int blocks = get_dim_size(scales, 1);
  for (int bb = begin[0]; bb < end[0]; ++bb) {
    int b0 = bb * 16;
    int b_end = s_i32_min(b0 + 16, blocks);
    for (int block = b0; block < b_end; ++block) {
      float64 mx = 0.0f;
      #pragma clang loop unroll_count(4)
      for (int g = 0; g < groups; ++g) {
        int5 pos = {g, block, 0, 0, 0};
        float64 sv = (float64)((uint64)v_f32_ld_g(gen_addr(pos, scales)) & 0x7fffffffu);
        mx = v_f32_max_b(mx, sv);
      }
      float64 out_s = fp8_pow2_ceil_scale(mx * (1.0f / 64.0f), mx);
      int5 dst = {0, block, 0, 0, 0};
      v_f32_st_tnsr_partial(dst, blk_scale, out_s, 0, 0);
    }
  }
}
