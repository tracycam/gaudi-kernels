// SPDX-License-Identifier: Apache-2.0
// Per-row activation residual scale (max(row_amax / 220.0f, 1e-30f)) for 2K FP8_143 cross term.
void main(tensor input, tensor row_scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int width = get_dim_size(input, 0);
  int rows = get_dim_size(input, 1);
  for (int rb = begin[0]; rb < end[0]; ++rb) {
    int r0 = rb * 16;
    int r_end = r0 + 16;
    if (r_end > rows) r_end = rows;
    for (int row = r0; row < r_end; ++row) {
      float64 maximum = 0.0f;
      #pragma clang loop unroll_count(4)
      for (int col = 0; col < width; col += 128) {
        int5 p = {col, row, 0, 0, 0};
        float128 x = convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(p, input), SW_LINEAR);
        float64 a = (float64)((uint64)x.v1 & 0x7fffffff);
        float64 b = (float64)((uint64)x.v2 & 0x7fffffff);
        maximum = v_f32_max_b(maximum, v_f32_max_b(a, b));
      }
      maximum = v_f32_reduce_max(maximum);
      float64 s = v_f32_max_b(maximum * (1.0f / 220.0f), 1e-30f);
      int5 dst = {0, row, 0, 0, 0};
      v_f32_st_tnsr_partial(dst, row_scale, s, 0, 0);
    }
  }
}
