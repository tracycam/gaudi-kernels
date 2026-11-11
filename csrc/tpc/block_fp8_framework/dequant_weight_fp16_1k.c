// SPDX-License-Identifier: Apache-2.0
#include "activation_quant_common.h"

// Emit SRAM-bundled weight operands from resident 1-byte FP8_143 weights:
// - default (1K FP16): w0 = FP16(w * eff_scale) in all_b lane order;
// - RESIDUAL_FP8_3K (3K FP8_143): [dw * 32, dw * 4, w0 / 128] packed 256 bytes
//   per group-pair so c0 + 4 * c1 reconstructs x0*w0 + x0*dw + dx*w0.
static inline minifloat256 fp8_143_pack256_hw(float128 lo, float128 hi) {
  float256 quad = {lo.v1, lo.v2, hi.v1, hi.v2};
  return v_convert_f32_to_f8_all_b(quad, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
}

void main(tensor input, tensor scales, tensor blk_scale, tensor output, int tile_tag) {
  if (tile_tag < 0) return;
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(input, 2);
  int b0 = begin[1];
  int b_end = end[1];
#ifdef RESIDUAL_FP8_3K
  int pairs = (groups + 1) / 2;
  int gp0 = begin[0];
  int gp_end = s_i32_min(end[0], pairs);
  for (int gp = gp0; gp < gp_end; ++gp) {
    int g0 = gp * 2;
    int base = gp * 768;
    bool has1 = (g0 + 1) < groups;
    for (int block = b0; block < b_end; ++block) {
      int row0 = block * 128;
      int5 bp = {0, block, 0, 0, 0};
      float64 inv_b = fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(bp, blk_scale)));
      int5 sp0 = {g0, block, 0, 0, 0};
      float64 eff0 = v_f32_ld_g(gen_addr(sp0, scales)) * inv_b;
      if (has1) {
        int5 sp1 = {g0 + 1, block, 0, 0, 0};
        float64 eff1 = v_f32_ld_g(gen_addr(sp1, scales)) * inv_b;
        #pragma clang loop unroll_count(2)
        for (int i = 0; i < 128; ++i) {
          int row = row0 + i;
          int5 src0 = {0, row, g0, 0, 0}, src1 = {0, row, g0 + 1, 0, 0};
          minifloat256 pk0 = v_f8_ld_tnsr_b(src0, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
          minifloat256 pk1 = v_f8_ld_tnsr_b(src1, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
          float128 ws0 = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(pk0));
          float128 ws1 = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(pk1));
          ws0.v1 *= eff0;
          ws0.v2 *= eff0;
          ws1.v1 *= eff1;
          ws1.v2 *= eff1;
          float128 w0_0 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(ws0, SW_RHNE));
          float128 w0_1 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(ws1, SW_RHNE));
          float128 dw0 = {ws0.v1 - w0_0.v1, ws0.v2 - w0_0.v2};
          float128 dw1 = {ws1.v1 - w0_1.v1, ws1.v2 - w0_1.v2};
          float128 bdw0_0 = {dw0.v1 * 32.0f, dw0.v2 * 32.0f};
          float128 bdw0_1 = {dw1.v1 * 32.0f, dw1.v2 * 32.0f};
          float128 bdw1_0 = {dw0.v1 * 4.0f, dw0.v2 * 4.0f};
          float128 bdw1_1 = {dw1.v1 * 4.0f, dw1.v2 * 4.0f};
          float128 bw0_0 = {w0_0.v1 * (1.0f / 128.0f), w0_0.v2 * (1.0f / 128.0f)};
          float128 bw0_1 = {w0_1.v1 * (1.0f / 128.0f), w0_1.v2 * (1.0f / 128.0f)};
          int5 d0 = {base, row, 0, 0, 0}, d1 = {base + 256, row, 0, 0, 0}, d2 = {base + 512, row, 0, 0, 0};
          v_f8_st_tnsr(d0, output, fp8_143_pack256_hw(bdw0_0, bdw0_1));
          v_f8_st_tnsr(d1, output, fp8_143_pack256_hw(bdw1_0, bdw1_1));
          v_f8_st_tnsr(d2, output, fp8_143_pack256_hw(bw0_0, bw0_1));
        }
      } else {
        float128 zero = {0.0f, 0.0f};
        #pragma clang loop unroll_count(2)
        for (int i = 0; i < 128; ++i) {
          int row = row0 + i;
          int5 src0 = {0, row, g0, 0, 0};
          minifloat256 pk0 = v_f8_ld_tnsr_b(src0, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
          float128 ws0 = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(pk0));
          ws0.v1 *= eff0;
          ws0.v2 *= eff0;
          float128 w0_0 = v_convert_f16_to_f32_all_b(convert_float128_to_half128(ws0, SW_RHNE));
          float128 dw0 = {ws0.v1 - w0_0.v1, ws0.v2 - w0_0.v2};
          float128 bdw0_0 = {dw0.v1 * 32.0f, dw0.v2 * 32.0f};
          float128 bdw1_0 = {dw0.v1 * 4.0f, dw0.v2 * 4.0f};
          float128 bw0_0 = {w0_0.v1 * (1.0f / 128.0f), w0_0.v2 * (1.0f / 128.0f)};
          int5 d0 = {base, row, 0, 0, 0}, d1 = {base + 256, row, 0, 0, 0}, d2 = {base + 512, row, 0, 0, 0};
          v_f8_st_tnsr(d0, output, fp8_143_pack256_hw(bdw0_0, zero));
          v_f8_st_tnsr(d1, output, fp8_143_pack256_hw(bdw1_0, zero));
          v_f8_st_tnsr(d2, output, fp8_143_pack256_hw(bw0_0, zero));
        }
      }
    }
  }
#else
  for (int group = begin[0]; group < end[0]; ++group) {
    int base = group * 128;
    for (int block = b0; block < b_end; ++block) {
      int row0 = block * 128;
      int5 scale_pos = {group, block, 0, 0, 0};
      float64 s_val = v_f32_ld_g(gen_addr(scale_pos, scales));
      int5 bp = {0, block, 0, 0, 0};
      float64 eff_scale = s_val * fp8_pow2_exact_rcp(v_f32_ld_g(gen_addr(bp, blk_scale)));
      #pragma clang loop unroll_count(2)
      for (int i = 0; i < 128; ++i) {
        int row = row0 + i;
        int5 source = {0, row, group, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(source, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, 1);
        float128 ws = v_convert_bf16_to_f32_all_b(v_convert_f8_to_bf16_b(packed));
        ws.v1 *= eff_scale;
        ws.v2 *= eff_scale;
        int5 dst = {base, row, 0, 0, 0};
        v_f16_st_tnsr(dst, output, convert_float128_to_half128(ws, SW_RHNE));
      }
    }
  }
#endif
}
