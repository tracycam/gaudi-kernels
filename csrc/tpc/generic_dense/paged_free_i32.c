// SPDX-License-Identifier: Apache-2.0
// Validates slot/page ranges and deduplicates freed pages via a caller-zeroed
// seen_bitset [bitset_words] device tensor and metadata [3] = {length, max_pages, bitset_words}.
void main(tensor indices, tensor metadata, tensor seen_bitset, tensor pages, tensor result_count,
          int page_size) {
  int5 point = {0, 0, 0, 0, 0};
  const int length = s_i32_ld_g(gen_addr(point, metadata));
  point[0] = 1;
  const int max_pages = s_i32_ld_g(gen_addr(point, metadata));
  point[0] = 2;
  const int bitset_words = s_i32_ld_g(gen_addr(point, metadata));
  int count = 0;
  for (int i = 0; i < length; ++i) {
    int5 index = {0, i, 0, 0, 0};
    const int slot = s_i32_ld_g(gen_addr(index, indices));
    index[0] = 1;
    const int high = s_i32_ld_g(gen_addr(index, indices));
    const int page = slot / page_size;
    const int word = page >> 5;
    if (high != 0 || slot < page_size || page > max_pages || word >= bitset_words) {
      count = -1;
      break;
    }
    const unsigned int mask = 1u << (page & 31);
    int5 wpos = {word, 0, 0, 0, 0};
    const unsigned int bits = (unsigned int)s_i32_ld_g(gen_addr(wpos, seen_bitset));
    if (!(bits & mask)) {
      s_i32_st_g(gen_addr(wpos, seen_bitset), (int)(bits | mask));
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
