// SPDX-License-Identifier: Apache-2.0
// Dedicated 1-output 128x128 block-FP8 weight producer for SRAM-bundled
// 1K FP16 + 2K FP8_143 GEMM (FP16 w0 when RESIDUAL_FP8_2K is unset,
// FP8_143 [32 * (ws - w0), w0 / 128] when RESIDUAL_FP8_2K is set).
#include "activation_quant_common.h"

void main(tensor input, tensor scales, tensor blk_scale, tensor output, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int group = begin[0]; group < end[0]; ++group) {
#ifdef RESIDUAL_FP8_2K
    int base = group * 256;
#else
    int base = group * 128;
#endif
    for (int block = begin[1]; block < end[1]; ++block) {
      int5 scale_pos = {group, block, 0, 0, 0};
      float64 s_val = v_f32_ld_g(gen_addr(scale_pos, scales));
      int5 bp = {0, block, 0, 0, 0};
      float64 eff_scale = s_val * fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(bp, blk_scale)));
#ifdef RESIDUAL_FP8_2K
      #pragma clang loop unroll_count(2)
#else
      #pragma clang loop unroll_count(4)
#endif
      for (int i = 0; i < 128; ++i) {
        int row = block * 128 + i;
        int5 source = {0, row, group, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(source, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
        float128 ws = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(packed));
        ws.v1 *= eff_scale;
        ws.v2 *= eff_scale;
        half128 h = convert_float128_to_half128(ws, SW_RHNE);
        int5 dst = {base, row, 0, 0, 0};
#ifdef RESIDUAL_FP8_2K
        float128 w0 = v_convert_f16_to_f32_all_b(h);
        float128 bdw = {(ws.v1 - w0.v1) * 32.0f, (ws.v2 - w0.v2) * 32.0f};
        float128 bw0 = {w0.v1 * (1.0f / 128.0f), w0.v2 * (1.0f / 128.0f)};
        v_f8_st_tnsr(dst, output, fp8_143_pack256_all_clamped(bdw, bw0));
#else
        v_f16_st_tnsr(dst, output, h);
#endif
      }
    }
  }
}
