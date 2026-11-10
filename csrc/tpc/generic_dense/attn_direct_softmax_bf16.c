// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Coarsened single-block causal softmax for KV buckets up to 512.
// Pairs safe exponentials and skips causally masked 128-key blocks within a
// 4 KiB VLM staging buffer (well inside the 16 KiB SFU/LUT VLM budget).
void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 h0 = {0, 0, 0, 0, 0}, h1 = {1, 0, 0, 0, 0}, h2 = {2, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(h0, header));
  const int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  const int key_start = s_i32_ld_g(gen_addr(h2, header));
  const int keys = get_dim_size(scores, 0), rows = get_dim_size(scores, 1);
  const int max_key_blocks = s_i32_max(0, s_i32_min(keys, (valid_keys + 127) & ~127));
  const uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;
  float64 buf[16];

  for (int head = begin[0]; head < end[0]; ++head) {
    for (int block = begin[1]; block < end[1]; ++block) {
      const int first = block * 16;
      const int last = s_i32_min(rows, first + 16);
      int row = first;

      if (keys == 512 && valid_keys == 512) {
        for (; row + 1 < last && row + 1 < valid_rows; row += 2) {
          int5 pc0 = {row, 0, 0, 0, 0}, pc1 = {row + 1, 0, 0, 0, 0};
          const int p0 = s_i32_ld_g(gen_addr(pc0, positions));
          const int p1 = s_i32_ld_g(gen_addr(pc1, positions));
          if (s_i32_min(p0, p1) - key_start < 511) break;

          int5 q0 = {0, row, head, 0, 0}, q1 = {0, row + 1, head, 0, 0};
          q0[0] = 0;   float64 a0 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 64;  float64 a1 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 128; float64 a2 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 192; float64 a3 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 256; float64 a4 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 320; float64 a5 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 384; float64 a6 = v_f32_ld_tnsr_b(q0, scores) * scale;
          q0[0] = 448; float64 a7 = v_f32_ld_tnsr_b(q0, scores) * scale;
          buf[0] = a0; buf[1] = a1; buf[2] = a2; buf[3] = a3;
          buf[4] = a4; buf[5] = a5; buf[6] = a6; buf[7] = a7;
          float64 m0 = v_f32_max_b(
              v_f32_max_b(v_f32_max_b(a0, a2), v_f32_max_b(a4, a6)),
              v_f32_max_b(v_f32_max_b(a1, a3), v_f32_max_b(a5, a7)));

          q1[0] = 0;   float64 b0 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 64;  float64 b1 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 128; float64 b2 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 192; float64 b3 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 256; float64 b4 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 320; float64 b5 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 384; float64 b6 = v_f32_ld_tnsr_b(q1, scores) * scale;
          q1[0] = 448; float64 b7 = v_f32_ld_tnsr_b(q1, scores) * scale;
          buf[8] = b0; buf[9] = b1; buf[10] = b2; buf[11] = b3;
          buf[12] = b4; buf[13] = b5; buf[14] = b6; buf[15] = b7;
          float64 m1 = v_f32_max_b(
              v_f32_max_b(v_f32_max_b(b0, b2), v_f32_max_b(b4, b6)),
              v_f32_max_b(v_f32_max_b(b1, b3), v_f32_max_b(b5, b7)));

          m0 = v_f32_reduce_max(m0);
          m1 = v_f32_reduce_max(m1);
          float64 o0 = v_f32_sel_eq_f32_b(m0, ni, 0.f, m0);
          float64 o1 = v_f32_sel_eq_f32_b(m1, ni, 0.f, m1);

          float64 sum0_a = 0.f, sum0_b = 0.f;
          #pragma clang loop unroll(disable)
          for (int k = 0; k < 8; k += 2) {
            float64 e0, e1;
            TPC_EXP_F32_SAFE_PAIR(buf[k] - o0, buf[k + 1] - o0, e0, e1);
            buf[k] = e0;
            buf[k + 1] = e1;
            sum0_a += e0;
            sum0_b += e1;
          }
          float64 sum1_a = 0.f, sum1_b = 0.f;
          #pragma clang loop unroll(disable)
          for (int k = 8; k < 16; k += 2) {
            float64 e0, e1;
            TPC_EXP_F32_SAFE_PAIR(buf[k] - o1, buf[k + 1] - o1, e0, e1);
            buf[k] = e0;
            buf[k + 1] = e1;
            sum1_a += e0;
            sum1_b += e1;
          }

          float64 s0 = v_f32_reduce_add(sum0_a + sum0_b);
          float64 s1 = v_f32_reduce_add(sum1_a + sum1_b);
          float64 inv0 = v_f32_sel_grt_f32_b(s0, 0.f, v_reciprocal_f32(v_f32_sel_grt_f32_b(s0, 0.f, s0, 1.f)), 0.f);
          float64 inv1 = v_f32_sel_grt_f32_b(s1, 0.f, v_reciprocal_f32(v_f32_sel_grt_f32_b(s1, 0.f, s1, 1.f)), 0.f);

          #pragma clang loop unroll(full)
          for (int k = 0; k < 4; ++k) {
            float128 p_row0 = {buf[2 * k] * inv0, buf[2 * k + 1] * inv0};
            float128 p_row1 = {buf[8 + 2 * k] * inv1, buf[8 + 2 * k + 1] * inv1};
            q0[0] = k * 128;
            q1[0] = k * 128;
            v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(p_row0, SW_LINEAR | SW_RHNE));
            v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(p_row1, SW_LINEAR | SW_RHNE));
          }
        }
      }

      for (; row < last; ++row) {
        int5 p = {0, row, head, 0, 0};
        if (row < valid_rows && max_key_blocks > 0) {
          int5 pc = {row, 0, 0, 0, 0};
          const int last_key_s = s_i32_min(s_i32_ld_g(gen_addr(pc, positions)) - key_start, valid_keys - 1);
          if (last_key_s >= 0) {
            const int active_blocks = s_i32_min(max_key_blocks, (last_key_s + 128) & ~127);
            const int64 last_key = last_key_s;
            float64 max0 = ni, max1 = ni;
            int slot = 0;
            #pragma clang loop unroll(disable)
            for (int i = 0; i < active_blocks; i += 128) {
              p[0] = i;      float64 a = v_f32_ld_tnsr_b(p, scores) * scale;
              p[0] = i + 64; float64 b = v_f32_ld_tnsr_b(p, scores) * scale;
              buf[slot] = a;
              buf[slot + 1] = b;
              max0 = v_f32_max_b(max0, v_f32_sel_leq_i32_b((int64)lanes, last_key - i, a, ni));
              max1 = v_f32_max_b(max1, v_f32_sel_leq_i32_b((int64)lanes, last_key - (i + 64), b, ni));
              slot += 2;
            }
            float64 maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
            float64 origin = v_f32_sel_eq_f32_b(maximum, ni, 0.f, maximum);
            float64 sum0 = 0.f, sum1 = 0.f;
            slot = 0;
            #pragma clang loop unroll(disable)
            for (int i = 0; i < active_blocks; i += 128) {
              float64 a, b;
              TPC_EXP_F32_SAFE_PAIR(buf[slot] - origin, buf[slot + 1] - origin, a, b);
              a = v_f32_sel_leq_i32_b((int64)lanes, last_key - i, a, 0.f);
              b = v_f32_sel_leq_i32_b((int64)lanes, last_key - (i + 64), b, 0.f);
              sum0 += a;
              sum1 += b;
              buf[slot] = a;
              buf[slot + 1] = b;
              slot += 2;
            }
            float64 sum = v_f32_reduce_add(sum0 + sum1);
            float64 safe = v_f32_sel_grt_f32_b(sum, 0.f, sum, 1.f);
            float64 inverse = v_f32_sel_grt_f32_b(sum, 0.f, v_reciprocal_f32(safe), 0.f);
            slot = 0;
            #pragma clang loop unroll(disable)
            for (int i = 0; i < active_blocks; i += 128) {
              float128 pair = {buf[slot] * inverse, buf[slot + 1] * inverse};
              p[0] = i;
              v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
              slot += 2;
            }
            for (int i = active_blocks; i < keys; i += 128) {
              p[0] = i;
              v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
            }
          } else {
            for (int i = 0; i < keys; i += 128) {
              p[0] = i;
              v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
            }
          }
        } else {
          for (int i = 0; i < keys; i += 128) {
            p[0] = i;
            v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
          }
        }
      }
    }
  }
}
