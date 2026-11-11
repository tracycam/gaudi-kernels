// SPDX-License-Identifier: Apache-2.0
// Combine primary BF16 GEMM accumulator c0, perturbed Karatsuba-32 GEMM accumulator cp,
// and FP32 bias via c0 + (cp - c0) / 32 + bias.
void main(tensor c0, tensor cp, tensor bias, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int blocks = get_dim_size(c0, 0) / 128;
  int rows = get_dim_size(c0, 1);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int bb = begin[0]; bb < end[0]; ++bb) {
      int b0 = bb * 4;
      int b_end = s_i32_min(b0 + 4, blocks);
      for (int bi = 0; bi < b_end - b0; ++bi) {
        int col = (b0 + bi) * 128;
        int5 bp0 = {col, 0, 0, 0, 0}, bp1 = {col + 64, 0, 0, 0, 0};
        float64 bv0 = v_f32_ld_tnsr_b(bp0, bias);
        float64 bv1 = v_f32_ld_tnsr_b(bp1, bias);
        #pragma clang loop unroll_count(2)
        for (int row = r0; row < r_end; ++row) {
          int5 p0 = {col, row, 0, 0, 0}, p1 = {col + 64, row, 0, 0, 0};
          float64 m0 = v_f32_ld_tnsr_b(p0, c0), m1 = v_f32_ld_tnsr_b(p1, c0);
          float64 p0v = v_f32_ld_tnsr_b(p0, cp), p1v = v_f32_ld_tnsr_b(p1, cp);
          float64 y0 = v_f32_mac_b(p0v - m0, 1.0f / 32.0f, m0) + bv0;
          float64 y1 = v_f32_mac_b(p1v - m1, 1.0f / 32.0f, m1) + bv1;
          float128 res = {y0, y1};
          v_bf16_st_tnsr(p0, output, convert_float128_to_bfloat128(res, SW_LINEAR | SW_RHNE));
        }
      }
    }
  }
}
