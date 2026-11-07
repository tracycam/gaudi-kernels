// SPDX-License-Identifier: Apache-2.0
// Source [heads * d, rows], destination [d, slots, heads]. The previous pools
// are graph inputs so Glue can require each sparse output to reuse its input.
void main(tensor key, tensor value, tensor locations, tensor metadata,
          tensor old_key_pool, tensor old_value_pool, tensor key_pool, tensor value_pool) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0,0,0,0,0};
  int valid = s_i32_ld_g(gen_addr(zero, metadata));
  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid) continue;
    int5 ip = {row,0,0,0,0};
    int slot = s_i32_ld_g(gen_addr(ip, locations));
    for (int head = begin[0]; head < end[0]; ++head) {
      int5 src = {head*128,row,0,0,0}, dst = {0,slot,head,0,0};
      bfloat128 k = v_bf16_ld_tnsr_b(src, key), v = v_bf16_ld_tnsr_b(src, value);
      v_bf16_st_tnsr(dst, key_pool, k); v_bf16_st_tnsr(dst, value_pool, v);
    }
  }
}
