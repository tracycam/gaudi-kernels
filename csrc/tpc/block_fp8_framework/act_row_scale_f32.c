// SPDX-License-Identifier: Apache-2.0
// Per-row power-of-2 activation scale (>= max_col |x[col, row]| / 16384, or 1.0f if row == 0)
// for SRAM-bundled 1K FP16 block-FP8 prefill.
#include "activation_quant_common.h"

void main(tensor input, tensor row_scale, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int width = get_dim_size(input, 0);
  int rows = get_dim_size(input, 1);
  for (int rb = begin[0]; rb < end[0]; ++rb) {
    int r0 = rb * 16;
    int r_end = s_i32_min(r0 + 16, rows);
    for (int row = r0; row < r_end; ++row) {
      float64 maximum = 0.0f;
      #pragma clang loop unroll_count(4)
      for (int col = 0; col < width; col += 128) {
        int5 p = {col, row, 0, 0, 0};
        float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, input));
        float64 a = (float64)((uint64)x.v1 & 0x7fffffffu);
        float64 b = (float64)((uint64)x.v2 & 0x7fffffffu);
        maximum = v_f32_max_b(maximum, v_f32_max_b(a, b));
      }
      maximum = v_f32_reduce_max(maximum);
      float64 s = fp8_pow2_ceil_scale(maximum * (1.0f / 16384.0f), maximum);
      int5 dst = {0, row, 0, 0, 0};
      v_f32_st_tnsr_partial(dst, row_scale, s, 0, 0);
    }
  }
}
