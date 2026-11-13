// SPDX-License-Identifier: Apache-2.0
// Head-major KV pool store for head_dim=256: coordinates are (kind * heads + head) * slots + slot.
void main(tensor key, tensor value, tensor locations, tensor metadata,
          tensor old_pool, tensor pool) {
  int5 zero = {0, 0, 0, 0, 0};
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int valid = s_i32_ld_g(gen_addr(zero, metadata));
  zero[0] = 1;
  int slots = s_i32_ld_g(gen_addr(zero, metadata));
  zero[0] = 2;
  int heads = s_i32_ld_g(gen_addr(zero, metadata));

  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid) continue;
    int5 ip = {row, 0, 0, 0, 0};
    int slot = s_i32_ld_g(gen_addr(ip, locations));
    for (int head = begin[0]; head < end[0]; ++head) {
      int5 src0 = {head * 256, row, 0, 0, 0};
      int5 src1 = {head * 256 + 128, row, 0, 0, 0};
      int5 dst0 = {0, head * slots + slot, 0, 0, 0};
      int5 dst1 = {128, head * slots + slot, 0, 0, 0};
      bfloat128 k0 = v_bf16_ld_tnsr_b(src0, key);
      bfloat128 k1 = v_bf16_ld_tnsr_b(src1, key);
      bfloat128 v0 = v_bf16_ld_tnsr_b(src0, value);
      bfloat128 v1 = v_bf16_ld_tnsr_b(src1, value);
      v_bf16_st_tnsr(dst0, pool, k0);
      v_bf16_st_tnsr(dst1, pool, k1);
      dst0[1] += heads * slots;
      dst1[1] += heads * slots;
      v_bf16_st_tnsr(dst0, pool, v0);
      v_bf16_st_tnsr(dst1, pool, v1);
    }
  }
}
