// SPDX-License-Identifier: Apache-2.0
// Dedicated 1-output 128x128 block-FP8 weight dequant producer for SRAM-bundled BF16 GEMM
// (w0 when KARATSUBA_32 is unset, w0 + 32 * (w_deq - w0) when set).
void main(tensor input, tensor scales, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int group = begin[0]; group < end[0]; ++group) {
    int base = group * 128;
    for (int block = begin[1]; block < end[1]; ++block) {
      int5 scale_pos = {group, block, 0, 0, 0};
      float scale = s_f32_ld_g(gen_addr(scale_pos, scales));
      #pragma clang loop unroll_count(4)
      for (int i = 0; i < 128; ++i) {
        int row = block * 128 + i;
        int5 source = {0, row, group, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(source, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
        float128 wide = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(packed));
        wide.v1 *= scale;
        wide.v2 *= scale;
        bfloat128 h = convert_float128_to_bfloat128(wide, SW_RHNE);
#ifdef KARATSUBA_32
        float128 hw = v_convert_bf16_to_f32_all_b(h);
        float128 wp = {v_f32_mac_b(wide.v1 - hw.v1, 32.0f, hw.v1),
                       v_f32_mac_b(wide.v2 - hw.v2, 32.0f, hw.v2)};
        h = convert_float128_to_bfloat128(wp, SW_RHNE);
#endif
        int5 dst = {base, row, 0, 0, 0};
        v_bf16_st_tnsr(dst, output, h);
      }
    }
  }
}
