// SPDX-License-Identifier: Apache-2.0

// Gather each request's causal KV prefix into head-major MME input tensors.
void main(tensor key_pool, tensor value_pool, tensor request_ids, tensor positions,
          tensor request_to_token, tensor metadata, tensor gathered_key, tensor gathered_value,
          int head_dim) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, metadata));
  for (int row = begin[2]; row < end[2]; ++row) {
    if (row >= valid_rows) continue;
    int5 index = {row, 0, 0, 0, 0};
    const int request = s_i32_ld_g(gen_addr(index, request_ids));
    const int position = s_i32_ld_g(gen_addr(index, positions));
    for (int head = begin[1]; head < end[1]; ++head) {
      for (int token = begin[0]; token < end[0]; ++token) {
        bfloat128 key = 0, value = 0;
        if (token <= position) {
          int5 map_pos = {token, request, 0, 0, 0};
          const int slot = s_i32_ld_g(gen_addr(map_pos, request_to_token));
          int5 source = {head * head_dim, slot, 0, 0, 0};
          key = v_bf16_ld_tnsr_partial_b(source, key_pool, (char)(head_dim - 1), 0);
          value = v_bf16_ld_tnsr_partial_b(source, value_pool, (char)(head_dim - 1), 0);
        }
        int5 target = {0, token, head, row, 0};
        v_bf16_st_tnsr_partial(target, gathered_key, key, (char)(head_dim - 1), 0);
        v_bf16_st_tnsr_partial(target, gathered_value, value, (char)(head_dim - 1), 0);
      }
    }
  }
}
