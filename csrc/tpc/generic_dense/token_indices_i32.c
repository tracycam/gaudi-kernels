// SPDX-License-Identifier: Apache-2.0
// Validate the active token IDs and zero the remainder of the static bucket.
void main(tensor tokens, tensor header, tensor indices, tensor status, int vocabulary) {
  const int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, header));
  const int lanes = get_dim_size(tokens, 0);
  for (int row = begin[0]; row < end[0]; ++row) {
    int5 output_pos = {row, 0, 0, 0, 0};
    int value = 0, error = 0;
    if (row < valid_rows) {
      int5 input_pos = {0, row, 0, 0, 0};
      value = s_i32_ld_g(gen_addr(input_pos, tokens));
      if (lanes == 2) {
        input_pos[0] = 1;
        if (s_i32_ld_g(gen_addr(input_pos, tokens)) != 0) error = 1;
      }
      if (value < 0 || value >= vocabulary) error = 1;
    }
    s_i32_st_g(gen_addr(output_pos, indices), error ? 0 : value);
    s_i32_st_g(gen_addr(output_pos, status), error);
  }
}
