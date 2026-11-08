// SPDX-License-Identifier: Apache-2.0
// The cache contains [cos[rotary_dim/2], sin[rotary_dim/2]] per position.
static inline void rotate_head(tensor input, tensor output, tensor cache, int row,
                               int head, int head_dim, int rotary_dim, int position,
                               int is_neox) {
  const int base = head * head_dim;
  const int rotary_half = rotary_dim / 2;
  if (is_neox) {
    for (int offset = 0; offset < rotary_half; offset += 128) {
      const int count = s_i32_min(rotary_half - offset, 128) - 1;
      int5 first = {base + offset, row, 0, 0, 0};
      int5 second = {base + rotary_half + offset, row, 0, 0, 0};
      int5 cosine = {offset, position, 0, 0, 0};
      int5 sine = {rotary_half + offset, position, 0, 0, 0};
      const float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(first, input, (char)count, 0));
      const float128 b = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(second, input, (char)count, 0));
      const float128 c = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(cosine, cache, (char)count, 0));
      const float128 s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(sine, cache, (char)count, 0));
      const float128 first_value = {a.v1 * c.v1 - b.v1 * s.v1, a.v2 * c.v2 - b.v2 * s.v2};
      const float128 second_value = {b.v1 * c.v1 + a.v1 * s.v1, b.v2 * c.v2 + a.v2 * s.v2};
      v_bf16_st_tnsr_partial(first, output, convert_float128_to_bfloat128(first_value, SW_RHNE), (char)count, 0);
      v_bf16_st_tnsr_partial(second, output, convert_float128_to_bfloat128(second_value, SW_RHNE), (char)count, 0);
    }
  } else {
    for (int offset = 0; offset < rotary_half; ++offset) {
      int5 first = {base + 2 * offset, row, 0, 0, 0};
      int5 second = {base + 2 * offset + 1, row, 0, 0, 0};
      int5 cosine = {offset, position, 0, 0, 0};
      int5 sine = {rotary_half + offset, position, 0, 0, 0};
      const float a = s_bf16_ld_g(gen_addr(first, input));
      const float b = s_bf16_ld_g(gen_addr(second, input));
      const float c = s_bf16_ld_g(gen_addr(cosine, cache));
      const float s = s_bf16_ld_g(gen_addr(sine, cache));
      const bf16 out_first = a * c - b * s, out_second = b * c + a * s;
      v_bf16_st_tnsr_partial(first, output, (bfloat128)out_first, 0, 0);
      v_bf16_st_tnsr_partial(second, output, (bfloat128)out_second, 0, 0);
    }
  }
  for (int offset = rotary_dim; offset < head_dim; offset += 128) {
    const int count = s_i32_min(head_dim - offset, 128) - 1;
    int5 point = {base + offset, row, 0, 0, 0};
    const bfloat128 unchanged = v_bf16_ld_tnsr_partial_b(point, input, (char)count, 0);
    v_bf16_st_tnsr_partial(point, output, unchanged, (char)count, 0);
  }
}

void main(tensor query, tensor key, tensor positions, tensor cache, tensor metadata,
          tensor query_out, tensor key_out, int head_dim, int rotary_dim, int is_neox) {
  const int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, metadata));
  const int q_heads = get_dim_size(query, 0) / head_dim;
  const int k_heads = get_dim_size(key, 0) / head_dim;
  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid_rows) continue;
    int5 index = {row, 0, 0, 0, 0};
    const int position = s_i32_ld_g(gen_addr(index, positions));
    for (int head = begin[0]; head < end[0]; ++head) {
      if (head < q_heads) rotate_head(query, query_out, cache, row, head, head_dim, rotary_dim, position, is_neox);
      if (head < k_heads) rotate_head(key, key_out, cache, row, head, head_dim, rotary_dim, position, is_neox);
    }
  }
}
