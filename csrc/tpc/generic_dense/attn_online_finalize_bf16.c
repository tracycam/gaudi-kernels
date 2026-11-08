// SPDX-License-Identifier: Apache-2.0
void main(tensor sums, tensor accumulator, tensor result) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(accumulator, 0), rows = get_dim_size(accumulator, 1);
  for (int head = begin[0]; head < end[0]; ++head) {
    for (int block = begin[1]; block < end[1]; ++block) {
      int5 stat = {block * 64,head,0,0,0};
      float64 denominator = v_f32_ld_tnsr_b(stat, sums);
      float64 safe = v_f32_sel_grt_f32_b(denominator, 0.f, denominator, 1.f);
      float64 inverse = v_f32_sel_grt_f32_b(denominator, 0.f, v_reciprocal_f32(safe), 0.f);
      int last = s_i32_min(rows - block * 64, 64);
      for (int item = 0; item < last; ++item) {
        float64 r = v_broadcast_element_f32(inverse, item);
        int5 p = {0,block * 64 + item,head,0,0};
        float64 a = v_f32_ld_tnsr_b(p, accumulator) * r, b = 0.f;
        if (width == 128) { p[0] = 64; b = v_f32_ld_tnsr_b(p, accumulator) * r; p[0] = 0; }
        float128 pair = {a,b};
        bfloat128 output = convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE);
        if (width == 128) v_bf16_st_tnsr(p, result, output);
        else v_bf16_st_tnsr_partial(p, result, output, 63, 0);
      }
    }
  }
}
