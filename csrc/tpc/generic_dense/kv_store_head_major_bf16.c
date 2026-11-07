// SPDX-License-Identifier: Apache-2.0
// Source [heads * d, rows]. K and V share one head-major pool: flattened
// coordinates are (kind * heads + head) * slots + slot. Metadata holds the
// valid row count, actual pool capacity, and head count, in that order.
void main(tensor key, tensor value, tensor locations, tensor metadata,
          tensor old_pool, tensor pool) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0,0,0,0,0};
  int valid = s_i32_ld_g(gen_addr(zero, metadata));
  zero[0] = 1;
  int slots = s_i32_ld_g(gen_addr(zero, metadata));
  zero[0] = 2;
  int heads = s_i32_ld_g(gen_addr(zero, metadata));
  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid) continue;
    int5 ip = {row,0,0,0,0};
    int slot = s_i32_ld_g(gen_addr(ip, locations));
    for (int head = begin[0]; head < end[0]; ++head) {
      int5 src = {head*128,row,0,0,0}, dst = {0,head*slots+slot,0,0,0};
      bfloat128 k = v_bf16_ld_tnsr_b(src, key), v = v_bf16_ld_tnsr_b(src, value);
      v_bf16_st_tnsr(dst, pool, k); dst[1] += heads * slots; v_bf16_st_tnsr(dst, pool, v);
    }
  }
}
