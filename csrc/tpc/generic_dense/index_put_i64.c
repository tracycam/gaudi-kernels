// SPDX-License-Identifier: Apache-2.0
// Indices are validated before launch. Update the addressed INT64 lanes only.
void main(tensor indices, tensor values, tensor output) {
  int5 begin = get_index_space_offset();
  if (begin[0] != 0) return;
  const int count = get_dim_size(indices, 0);
  for (int row = 0; row < count; ++row) {
    int5 index = {row, 0, 0, 0, 0};
    int slot = s_i32_ld_g(gen_addr(index, indices));
    int5 source = {0, row, 0, 0, 0}, destination = {0, slot, 0, 0, 0};
    int low = s_i32_ld_g(gen_addr(source, values));
    source[0] = 1;
    int high = s_i32_ld_g(gen_addr(source, values));
    s_i32_st_g(gen_addr(destination, output), low);
    destination[0] = 1;
    s_i32_st_g(gen_addr(destination, output), high);
  }
}
