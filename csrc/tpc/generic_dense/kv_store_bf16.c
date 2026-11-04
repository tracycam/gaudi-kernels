// SPDX-License-Identifier: Apache-2.0
// NHD KV rows are addressed by the SGLang physical slot map.
void main(tensor key, tensor value, tensor locations, tensor metadata,
          tensor key_pool, tensor value_pool) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 header = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(header, metadata));
  const int width = get_dim_size(key, 0);
  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid_rows) continue;
    int5 location = {row, 0, 0, 0, 0};
    const int slot = s_i32_ld_g(gen_addr(location, locations));
    for (int block = begin[0]; block < end[0]; ++block) {
      int5 source = {block * 128, row, 0, 0, 0};
      int5 target = {block * 128, slot, 0, 0, 0};
      int count = s_i32_min(width - source[0], 128) - 1;
      bfloat128 k = v_bf16_ld_tnsr_partial_b(source, key, (char)count, 0);
      bfloat128 v = v_bf16_ld_tnsr_partial_b(source, value, (char)count, 0);
      v_bf16_st_tnsr_partial(target, key_pool, k, (char)count, 0);
      v_bf16_st_tnsr_partial(target, value_pool, v, (char)count, 0);
    }
  }
}
