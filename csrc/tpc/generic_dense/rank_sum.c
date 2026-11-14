// SPDX-License-Identifier: Apache-2.0
// Reduce world_size contiguous 1D rank slices [W * elems] -> [elems] in canonical rank order.

void main(tensor gathered, tensor dst, int world_size) {
  const int elems = get_dim_size(dst, 0);
#ifdef RANK_SUM_BF16
  const int step = 128;
#else
  const int step = 64;
#endif
  const int chunks = (elems + step - 1) / step;
  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const int per = chunks / ntid;
  const int rem = chunks % ntid;
  const int start = tid * per + s_i32_min(tid, rem);
  const int end = start + per + (tid < rem ? 1 : 0);

  for (int c = start; c < end; ++c) {
    const int col = c * step;
    const char cnt = (char)(s_i32_min(step, elems - col) - 1);
#ifdef RANK_SUM_BF16
    float128 acc = convert_bfloat128_to_float128(
        v_bf16_ld_tnsr_partial_b((int5){col, 0, 0, 0, 0}, gathered, cnt, 0), SW_LINEAR);
    for (int r = 1; r < world_size; ++r) {
      float128 v = convert_bfloat128_to_float128(
          v_bf16_ld_tnsr_partial_b((int5){r * elems + col, 0, 0, 0, 0}, gathered, cnt, 0),
          SW_LINEAR);
      acc.v1 = acc.v1 + v.v1;
      acc.v2 = acc.v2 + v.v2;
    }
    v_bf16_st_tnsr_partial((int5){col, 0, 0, 0, 0}, dst,
                           convert_float128_to_bfloat128(acc, SW_LINEAR | SW_RHNE), cnt, 0);
#else
    float64 acc = v_f32_ld_tnsr_partial_b((int5){col, 0, 0, 0, 0}, gathered, cnt, 0);
    for (int r = 1; r < world_size; ++r) {
      acc = acc + v_f32_ld_tnsr_partial_b((int5){r * elems + col, 0, 0, 0, 0}, gathered, cnt, 0);
    }
    v_f32_st_tnsr_partial((int5){col, 0, 0, 0, 0}, dst, acc, cnt, 0);
#endif
  }
}
