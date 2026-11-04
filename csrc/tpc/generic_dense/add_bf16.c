// SPDX-License-Identifier: Apache-2.0
// Consume materialized BF16 values and round their FP32 sum to BF16.
void main(tensor left, tensor right, tensor output) {
  const int width = get_dim_size(left, 0);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int block = begin[0]; block < end[0]; ++block) {
      int5 coord = {block * 128, row, 0, 0, 0};
      int count = s_i32_min(width - coord[0], 128) - 1;
      float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(coord, left, (char)count, 0));
      float128 b = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(coord, right, (char)count, 0));
      float128 sum = {a.v1 + b.v1, a.v2 + b.v2};
      v_bf16_st_tnsr_partial(coord, output, convert_float128_to_bfloat128(sum, SW_RHNE), (char)count, 0);
    }
  }
}
