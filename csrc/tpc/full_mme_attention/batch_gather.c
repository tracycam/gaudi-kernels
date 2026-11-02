// SPDX-License-Identifier: Apache-2.0
// Materialize each request's physical pages into its own MME batch plane.
void main(tensor key, tensor value, tensor page_ids, tensor usage, tensor groups,
          tensor row_groups, tensor key_out, tensor value_out) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int physical_pages = get_dim_size(key, 2) / 128;
  for (int row = begin[1]; row < end[1]; ++row) {
    int5 rp = {row, 0, 0, 0, 0};
    int row_group = s_i32_ld_g(gen_addr(rp, row_groups));
    for (int page = begin[0]; page < end[0]; ++page) {
      int5 index = {page, row, 0, 0, 0};
      int physical = s_i32_ld_g(gen_addr(index, page_ids));
      int count = s_i32_ld_g(gen_addr(index, usage));
      int group = s_i32_ld_g(gen_addr(index, groups));
      int valid = row_group >= 0 && group == row_group && physical >= 0 && physical < physical_pages;
      int high = valid ? s_i32_max(0, s_i32_min(count, 128)) : 0;
      for (int offset = 0; offset < 128; ++offset) {
        bfloat128 key0 = 0, key1 = 0, val = 0;
        if (offset < high) {
          int5 src = {0, 0, physical * 128 + offset, 0, 0};
          key0 = v_bf16_ld_tnsr_b(src, key);
          src[0] = 128;
          key1 = v_bf16_ld_tnsr_partial_b(src, key, 63, 0);
          src[0] = 0;
          val = v_bf16_ld_tnsr_b(src, value);
        }
        int5 dst = {0, page * 128 + offset, row, 0, 0};
        v_bf16_st_tnsr(dst, key_out, key0);
        dst[0] = 128;
        v_bf16_st_tnsr_partial(dst, key_out, key1, 63, 0);
        dst[0] = 0;
        v_bf16_st_tnsr(dst, value_out, val);
      }
    }
  }
}
