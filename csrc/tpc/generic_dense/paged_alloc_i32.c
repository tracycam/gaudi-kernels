// SPDX-License-Identifier: Apache-2.0
// Device slot assignment from page IDs and host-prepared request lengths.
void main(tensor last_slots, tensor pages, tensor metadata, tensor result, int page_size) {
  const int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int req = begin[0]; req < end[0]; ++req) {
    int5 at = {0, req, 0, 0, 0};
    const int prefix = s_i32_ld_g(gen_addr(at, metadata));
    at[0] = 1;
    const int length = s_i32_ld_g(gen_addr(at, metadata));
    at[0] = 2;
    const int first_out = s_i32_ld_g(gen_addr(at, metadata));
    at[0] = 3;
    const int first_page = s_i32_ld_g(gen_addr(at, metadata));
    const int first_count = s_i32_min(length, (page_size - prefix % page_size) % page_size);
    int last = -1;
    if (first_count) {
      int5 previous = {0, req, 0, 0, 0};
      last = s_i32_ld_g(gen_addr(previous, last_slots));
    }
    for (int i = 0; i < length; ++i) {
      int slot;
      if (i < first_count) {
        slot = last + i + 1;
      } else {
        const int offset = i - first_count;
        int5 page = {0, first_page + offset / page_size, 0, 0, 0};
        slot = s_i32_ld_g(gen_addr(page, pages)) * page_size + offset % page_size;
      }
      int5 out = {0, first_out + i, 0, 0, 0};
      s_i32_st_g(gen_addr(out, result), slot);
      out[0] = 1;
      s_i32_st_g(gen_addr(out, result), 0);
    }
  }
}
