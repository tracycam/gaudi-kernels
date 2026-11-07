// SPDX-License-Identifier: Apache-2.0
// The 64-element head uses a half-vector store; Glue requires the old pool to
// alias the sparse output, so other slots and the adjacent head stay intact.
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
      int5 src = {head*64,row,0,0,0}, dst = {0,slot,head,0,0};
      bfloat128 k = v_bf16_ld_tnsr_partial_b(src, key, 63, 0);
      bfloat128 v = v_bf16_ld_tnsr_partial_b(src, value, 63, 0);
      v_bf16_st_tnsr_partial(dst, key_pool, k, 63, 0);
      v_bf16_st_tnsr_partial(dst, value_pool, v, 63, 0);
    }
  }
}
