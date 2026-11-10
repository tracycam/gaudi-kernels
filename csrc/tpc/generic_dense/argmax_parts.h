// SPDX-License-Identifier: Apache-2.0
// One vocabulary pass, retaining PyTorch's first-NaN and first-tie behavior.
void main(tensor input, tensor maxima, tensor indices, tensor nan_indices,
          int width, int blocks_per_part) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const uint64 lane = read_lane_id_4b_b();
  const float negative_infinity = -1.f / 0.f;
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int part = begin[0]; part < end[0]; ++part) {
      float64 best = negative_infinity;
      int64 best_index = 2147483647, first_nan = 2147483647;
      const int first = part * blocks_per_part;
      const int last = s_i32_min(first + blocks_per_part, (width + 127) / 128);
      for (int block = first; block < last; ++block) {
        int5 position = {block * 128, row, 0, 0, 0};
#ifdef ARGMAX_BF16
        float128 pair = convert_bfloat128_to_float128(v_bf16_ld_tnsr_b(position, input), SW_LINEAR);
        float64 left = pair.v1, right = pair.v2;
#else
        float64 left = v_f32_ld_tnsr_b(position, input);
        position[0] += 64;
        float64 right = v_f32_ld_tnsr_b(position, input);
#endif
#define UPDATE(value, offset) do { \
        uint64 candidate = lane + (unsigned)(block * 128 + (offset)); \
        int64 valid_index = v_i32_sel_less_u32_b(candidate, (unsigned)width, (int64)candidate, 2147483647); \
        uint64 magnitude = (uint64)(value) & 0x7fffffff; \
        int64 nan_index = v_i32_sel_grt_u32_b(magnitude, 0x7f800000, valid_index, 2147483647); \
        first_nan = v_i32_min_b(first_nan, nan_index); \
        float64 clean = v_f32_sel_grt_u32_b(magnitude, 0x7f800000, negative_infinity, (value)); \
        clean = v_f32_sel_less_u32_b(candidate, (unsigned)width, clean, negative_infinity); \
        best_index = v_i32_sel_grt_f32_b(clean, best, valid_index, best_index); \
        best_index = v_i32_sel_eq_f32_b(clean, best, v_i32_min_b(valid_index, best_index), best_index); \
        best = v_f32_max_b(best, clean); \
      } while (0)
        UPDATE(left, 0);
        UPDATE(right, 64);
#undef UPDATE
      }
      float64 maximum = v_f32_reduce_max(best);
      best_index = v_i32_reduce_min(v_i32_sel_eq_f32_b(best, maximum, best_index, 2147483647));
      first_nan = v_i32_reduce_min(first_nan);
      int5 output = {0, part, row, 0, 0};
      v_f32_st_tnsr(output, maxima, maximum);
      v_i32_st_tnsr(output, indices, best_index);
      v_i32_st_tnsr(output, nan_indices, first_nan);
    }
  }
}
