// SPDX-License-Identifier: Apache-2.0
// Per-128-column block scale (max(max_g(scales[g, block]) * 1.02f, 1e-30f)) for 2K FP8_143 weights.
void main(tensor scales, tensor blk_scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(scales, 0);
  int blocks = get_dim_size(scales, 1);
  for (int bb = begin[0]; bb < end[0]; ++bb) {
    int b0 = bb * 16;
    int b_end = b0 + 16;
    if (b_end > blocks) b_end = blocks;
    for (int block = b0; block < b_end; ++block) {
      float64 mx = 0.0f;
      #pragma clang loop unroll_count(4)
      for (int g = 0; g < groups; ++g) {
        int5 pos = {g, block, 0, 0, 0};
        mx = v_f32_max_b(mx, v_f32_ld_g(gen_addr(pos, scales)));
      }
      float64 out_s = v_f32_max_b(mx * 1.02f, 1e-30f);
      int5 dst = {0, block, 0, 0, 0};
      v_f32_st_tnsr_partial(dst, blk_scale, out_s, 0, 0);
    }
  }
}
