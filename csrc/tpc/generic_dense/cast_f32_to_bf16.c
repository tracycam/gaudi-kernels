// SPDX-License-Identifier: Apache-2.0
// Materialize the required BF16 rounding boundary inside a larger graph.
void main(tensor input, tensor output) {
  const int width = get_dim_size(input, 0);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int block = begin[0]; block < end[0]; ++block) {
      int5 pos = {block * 128, row, 0, 0, 0};
      float128 values;
      values.v1 = v_f32_ld_tnsr_b(pos, input);
      pos[0] += 64;
      values.v2 = v_f32_ld_tnsr_b(pos, input);
      pos[0] -= 64;
      int count = s_i32_min(width - pos[0], 128) - 1;
      v_bf16_st_tnsr_partial(pos, output, convert_float128_to_bfloat128(values, SW_LINEAR | SW_RHNE),
                             (char)count, 0);
    }
  }
}
