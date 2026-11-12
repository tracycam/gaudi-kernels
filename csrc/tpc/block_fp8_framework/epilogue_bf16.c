// SPDX-License-Identifier: Apache-2.0
// Scale FP32 normalized 1K FP16 GEMM accumulator (c0) via (c0 * row_scale) * blk_scale into BF16.
void main(tensor c0, tensor row_scale, tensor blk_scale, tensor output) {
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
        int block = b0 + bi;
        int5 bp = {0, block, 0, 0, 0};
        float bscale = s_f32_ld_g(gen_addr(bp, blk_scale));
        int col = block * 128;
        #pragma clang loop unroll_count(2)
        for (int row = r0; row < r_end; ++row) {
          int5 rp = {0, row, 0, 0, 0};
          float rscale = s_f32_ld_g(gen_addr(rp, row_scale));
          int5 p0 = {col, row, 0, 0, 0}, p1 = {col + 64, row, 0, 0, 0};
          float64 m0 = v_f32_ld_tnsr_b(p0, c0), m1 = v_f32_ld_tnsr_b(p1, c0);
          float128 res = {(m0 * rscale) * bscale, (m1 * rscale) * bscale};
          v_bf16_st_tnsr(p0, output, convert_float128_to_bfloat128(res, SW_LINEAR | SW_RHNE));
        }
      }
    }
  }
}
