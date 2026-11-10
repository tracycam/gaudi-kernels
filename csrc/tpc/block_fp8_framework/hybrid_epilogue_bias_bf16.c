// SPDX-License-Identifier: Apache-2.0
// Combine FP32 main GEMM (c0), scaled FP8_143 cross-residual GEMM (c1), and FP32 bias into BF16.
void main(tensor c0, tensor c1, tensor row_scale, tensor blk_scale, tensor bias, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int blocks = get_dim_size(c0, 0) / 128;
  int rows = get_dim_size(c0, 1);
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    int r0 = rb * 8;
    int r_end = s_i32_min(r0 + 8, rows);
    for (int bb = begin[0]; bb < end[0]; ++bb) {
      int b0 = bb * 4;
      int b_end = s_i32_min(b0 + 4, blocks);
      float64 bscales[4], bv0[4], bv1[4];
      for (int bi = 0; bi < b_end - b0; ++bi) {
        int block = b0 + bi;
        int5 bp = {0, block, 0, 0, 0};
        bscales[bi] = v_f32_ld_g(gen_addr(bp, blk_scale));
        int col = block * 128;
        int5 b0_pos = {col, 0, 0, 0, 0}, b1_pos = {col + 64, 0, 0, 0, 0};
        bv0[bi] = v_f32_ld_tnsr_b(b0_pos, bias);
        bv1[bi] = v_f32_ld_tnsr_b(b1_pos, bias);
      }
      for (int row = r0; row < r_end; ++row) {
        int5 rp = {0, row, 0, 0, 0};
        float64 rscale = v_f32_ld_g(gen_addr(rp, row_scale)) * (1.0f / 256.0f);
        #pragma clang loop unroll_count(2)
        for (int bi = 0; bi < b_end - b0; ++bi) {
          float64 scale = rscale * bscales[bi];
          int col = (b0 + bi) * 128;
          int5 p0 = {col, row, 0, 0, 0}, p1 = {col + 64, row, 0, 0, 0};
          float64 m0 = v_f32_ld_tnsr_b(p0, c0) + bv0[bi];
          float64 m1 = v_f32_ld_tnsr_b(p1, c0) + bv1[bi];
          float64 r0_v = v_f32_ld_tnsr_b(p0, c1), r1_v = v_f32_ld_tnsr_b(p1, c1);
          float128 res = {v_f32_mac_b(r0_v, scale, m0), v_f32_mac_b(r1_v, scale, m1)};
          v_bf16_st_tnsr(p0, output, convert_float128_to_bfloat128(res, SW_LINEAR | SW_RHNE));
        }
      }
    }
  }
}
