// SPDX-License-Identifier: Apache-2.0
// Deduplicate released physical page IDs without copying slot indices to host.
__local unsigned int seen_pages[257];

void main(tensor indices, tensor metadata, tensor pages, tensor result_count, int page_size) {
  for (int i = 0; i < 257; ++i) seen_pages[i] = 0;
  int5 point = {0, 0, 0, 0, 0};
  const int length = s_i32_ld_g(gen_addr(point, metadata));
  point[0] = 1;
  const int max_pages = s_i32_ld_g(gen_addr(point, metadata));
  int count = 0;
  for (int i = 0; i < length; ++i) {
    int5 index = {0, i, 0, 0, 0};
    const int slot = s_i32_ld_g(gen_addr(index, indices));
    index[0] = 1;
    const int high = s_i32_ld_g(gen_addr(index, indices));
    const int page = slot / page_size;
    if (high != 0 || slot < page_size || page > max_pages) {
      count = -1;
      break;
    }
    const int word = page >> 5;
    const unsigned int mask = 1u << (page & 31);
    if (!(seen_pages[word] & mask)) {
      seen_pages[word] |= mask;
      int5 out = {0, count, 0, 0, 0};
      s_i32_st_g(gen_addr(out, pages), page);
      out[0] = 1;
      s_i32_st_g(gen_addr(out, pages), 0);
      ++count;
    }
  }
  point[0] = 0;
  s_i32_st_g(gen_addr(point, result_count), count);
}
