// SPDX-License-Identifier: Apache-2.0
void main(tensor maxima, tensor indices, tensor nan_indices, tensor result, int parts) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[0]; row < end[0]; ++row) {
    float maximum = -1.f / 0.f;
    int index = 2147483647, first_nan = 2147483647;
    for (int part = 0; part < parts; ++part) {
      int5 at = {0, part, row, 0, 0};
      float value = s_f32_ld_g(gen_addr(at, maxima));
      int candidate = s_i32_ld_g(gen_addr(at, indices));
      int nan = s_i32_ld_g(gen_addr(at, nan_indices));
      if (value > maximum || (value == maximum && candidate < index)) {
        maximum = value;
        index = candidate;
      }
      first_nan = s_i32_min(first_nan, nan);
    }
    int5 out = {0, row, 0, 0, 0};
    s_i32_st_g(gen_addr(out, result), first_nan == 2147483647 ? index : first_nan);
  }
}
