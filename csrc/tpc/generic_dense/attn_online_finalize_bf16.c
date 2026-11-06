// SPDX-License-Identifier: Apache-2.0
// The final divide is performed on device; an empty row returns zero.
void main(tensor sums, tensor accumulator, tensor result) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(accumulator, 0);
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int row = begin[1]; row < end[1]; ++row) {
      int5 stat = {row, head, 0, 0, 0};
      float sum = s_f32_ld_g(gen_addr(stat, sums));
      float64 reciprocal = sum > 0.f ? v_reciprocal_f32((float64)sum) : 0.f;
      for (int col = 0; col < width; col += 128) {
        int5 p = {col, row, head, 0, 0};
        int lo = s_i32_min(width - col, 64);
        int hi = s_i32_max(0, s_i32_min(width - col - 64, 64));
        float64 a = v_f32_ld_tnsr_partial_b(p, accumulator, (char)(lo - 1), 0);
        p[0] += 64;
        float64 b = hi ? v_f32_ld_tnsr_partial_b(p, accumulator, (char)(hi - 1), 0) : (float64)0.f;
        float128 pair = {a * reciprocal, b * reciprocal};
        p[0] -= 64;
        int count = s_i32_min(width - col, 128) - 1;
        v_bf16_st_tnsr_partial(p, result, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE), (char)count, 0);
      }
    }
  }
}
