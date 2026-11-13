// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Coarsened 2-row LUT-free causal tile softmax for 3D {keys, rows, heads} and
// 4D {keys, rows, group, kv_heads} KV buckets up to 512.
// Uses 2-FMA Cody-Waite degree-6 Estrin expf and 3-step Newton-Raphson rcp (0
// SFU LUTs, full 64 KiB VLM unlocked) with 2-row tensor-load pipelining.
void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, tensor tile_max, tensor tile_sum,
          float scale, int block_rows, float softcap, int sliding_window) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 h0 = {0, 0, 0, 0, 0}, h1 = {1, 0, 0, 0, 0}, h2 = {2, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(h0, header));
  const int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  const int key_start = s_i32_ld_g(gen_addr(h2, header));
  const int keys = get_dim_size(scores, 0), rows = get_dim_size(scores, 1);
  const int max_key_blocks = s_i32_max(0, s_i32_min(keys, (valid_keys + 127) & ~127));
  const uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;
  const int plain_causal = (softcap <= 0.0f && sliding_window <= 0);

  if (plain_causal) {
  for (int kv_h = begin[3]; kv_h < end[3]; ++kv_h) {
    for (int head = begin[0]; head < end[0]; ++head) {
      for (int block = begin[1]; block < end[1]; ++block) {
        const int first = block * block_rows;
        const int last = s_i32_min(rows, first + block_rows);
        float64 acc_max = ni, acc_sum = 0.f;
        int row = first;

        if (max_key_blocks > 0) {
          for (; row + 1 < last && row + 1 < valid_rows; row += 2) {
            int5 pc0 = {row, 0, 0, 0, 0}, pc1 = {row + 1, 0, 0, 0, 0};
            const int lk0 = s_i32_min(s_i32_ld_g(gen_addr(pc0, positions)) - key_start, valid_keys - 1);
            const int lk1 = s_i32_min(s_i32_ld_g(gen_addr(pc1, positions)) - key_start, valid_keys - 1);
            if (lk0 < 0 || lk1 < 0) break;

            int5 q0 = {0, row, head, kv_h, 0}, q1 = {0, row + 1, head, kv_h, 0};
            if (keys == 512 && lk0 >= 511 && lk1 >= 511) {
              q0[0] = 0;   float64 a0 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 64;  float64 a1 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 128; float64 a2 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 192; float64 a3 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 256; float64 a4 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 320; float64 a5 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 384; float64 a6 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 448; float64 a7 = v_f32_ld_tnsr_b(q0, scores) * scale;

              q1[0] = 0;   float64 b0 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 64;  float64 b1 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 128; float64 b2 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 192; float64 b3 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 256; float64 b4 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 320; float64 b5 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 384; float64 b6 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 448; float64 b7 = v_f32_ld_tnsr_b(q1, scores) * scale;

              float64 m0 = v_f32_reduce_max(v_f32_max_b(
                  v_f32_max_b(v_f32_max_b(a0, a2), v_f32_max_b(a4, a6)),
                  v_f32_max_b(v_f32_max_b(a1, a3), v_f32_max_b(a5, a7))));
              float64 m1 = v_f32_reduce_max(v_f32_max_b(
                  v_f32_max_b(v_f32_max_b(b0, b2), v_f32_max_b(b4, b6)),
                  v_f32_max_b(v_f32_max_b(b1, b3), v_f32_max_b(b5, b7))));
              float64 o0 = v_f32_sel_eq_f32_b(m0, ni, 0.f, m0);
              float64 o1 = v_f32_sel_eq_f32_b(m1, ni, 0.f, m1);

              TPC_EXP_NEG_F32_PAIR(a0 - o0, a1 - o0, a0, a1);
              float64 sum0a = a0, sum0b = a1;
              TPC_EXP_NEG_F32_PAIR(b0 - o1, b1 - o1, b0, b1);
              float64 sum1a = b0, sum1b = b1;
              TPC_EXP_NEG_F32_PAIR(a2 - o0, a3 - o0, a2, a3);
              sum0a += a2; sum0b += a3;
              TPC_EXP_NEG_F32_PAIR(b2 - o1, b3 - o1, b2, b3);
              sum1a += b2; sum1b += b3;
              TPC_EXP_NEG_F32_PAIR(a4 - o0, a5 - o0, a4, a5);
              sum0a += a4; sum0b += a5;
              TPC_EXP_NEG_F32_PAIR(b4 - o1, b5 - o1, b4, b5);
              sum1a += b4; sum1b += b5;
              TPC_EXP_NEG_F32_PAIR(a6 - o0, a7 - o0, a6, a7);
              sum0a += a6; sum0b += a7;
              TPC_EXP_NEG_F32_PAIR(b6 - o1, b7 - o1, b6, b7);
              sum1a += b6; sum1b += b7;

              float64 s0 = v_f32_reduce_add(sum0a + sum0b);
              float64 s1 = v_f32_reduce_add(sum1a + sum1b);
              const int item0 = row - first;
              acc_max = v_f32_sel_eq_u32_b(lanes, item0, m0, acc_max);
              acc_max = v_f32_sel_eq_u32_b(lanes, item0 + 1, m1, acc_max);
              acc_sum = v_f32_sel_eq_u32_b(lanes, item0, s0, acc_sum);
              acc_sum = v_f32_sel_eq_u32_b(lanes, item0 + 1, s1, acc_sum);

              float64 inv0 = tpc_rcp_nonneg_f32(s0);
              float64 inv1 = tpc_rcp_nonneg_f32(s1);
              float128 pa0 = {a0 * inv0, a1 * inv0}, pb0 = {b0 * inv1, b1 * inv1};
              float128 pa1 = {a2 * inv0, a3 * inv0}, pb1 = {b2 * inv1, b3 * inv1};
              float128 pa2 = {a4 * inv0, a5 * inv0}, pb2 = {b4 * inv1, b5 * inv1};
              float128 pa3 = {a6 * inv0, a7 * inv0}, pb3 = {b6 * inv1, b7 * inv1};
              q0[0] = 0;   v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa0, SW_LINEAR | SW_RHNE));
              q1[0] = 0;   v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb0, SW_LINEAR | SW_RHNE));
              q0[0] = 128; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa1, SW_LINEAR | SW_RHNE));
              q1[0] = 128; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb1, SW_LINEAR | SW_RHNE));
              q0[0] = 256; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa2, SW_LINEAR | SW_RHNE));
              q1[0] = 256; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb2, SW_LINEAR | SW_RHNE));
              q0[0] = 384; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa3, SW_LINEAR | SW_RHNE));
              q1[0] = 384; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb3, SW_LINEAR | SW_RHNE));
            } else {
              const int ab = s_i32_max(s_i32_min(max_key_blocks, (lk0 + 128) & ~127),
                                       s_i32_min(max_key_blocks, (lk1 + 128) & ~127));
              const int64 vlk0 = lk0, vlk1 = lk1;
              q0[0] = 0;  float64 a0 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q0[0] = 64; float64 a1 = v_f32_ld_tnsr_b(q0, scores) * scale;
              q1[0] = 0;  float64 b0 = v_f32_ld_tnsr_b(q1, scores) * scale;
              q1[0] = 64; float64 b1 = v_f32_ld_tnsr_b(q1, scores) * scale;
              float64 max0a = v_f32_sel_leq_i32_b((int64)lanes, vlk0, a0, ni);
              float64 max0b = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 64, a1, ni);
              float64 max1a = v_f32_sel_leq_i32_b((int64)lanes, vlk1, b0, ni);
              float64 max1b = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 64, b1, ni);

              float64 a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
              float64 b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f, b6 = 0.f, b7 = 0.f;
              if (ab > 128) {
                q0[0] = 128; a2 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q0[0] = 192; a3 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q1[0] = 128; b2 = v_f32_ld_tnsr_b(q1, scores) * scale;
                q1[0] = 192; b3 = v_f32_ld_tnsr_b(q1, scores) * scale;
                max0a = v_f32_max_b(max0a, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 128, a2, ni));
                max0b = v_f32_max_b(max0b, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 192, a3, ni));
                max1a = v_f32_max_b(max1a, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 128, b2, ni));
                max1b = v_f32_max_b(max1b, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 192, b3, ni));
              }
              if (ab > 256) {
                q0[0] = 256; a4 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q0[0] = 320; a5 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q1[0] = 256; b4 = v_f32_ld_tnsr_b(q1, scores) * scale;
                q1[0] = 320; b5 = v_f32_ld_tnsr_b(q1, scores) * scale;
                max0a = v_f32_max_b(max0a, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 256, a4, ni));
                max0b = v_f32_max_b(max0b, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 320, a5, ni));
                max1a = v_f32_max_b(max1a, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 256, b4, ni));
                max1b = v_f32_max_b(max1b, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 320, b5, ni));
              }
              if (ab > 384) {
                q0[0] = 384; a6 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q0[0] = 448; a7 = v_f32_ld_tnsr_b(q0, scores) * scale;
                q1[0] = 384; b6 = v_f32_ld_tnsr_b(q1, scores) * scale;
                q1[0] = 448; b7 = v_f32_ld_tnsr_b(q1, scores) * scale;
                max0a = v_f32_max_b(max0a, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 384, a6, ni));
                max0b = v_f32_max_b(max0b, v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 448, a7, ni));
                max1a = v_f32_max_b(max1a, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 384, b6, ni));
                max1b = v_f32_max_b(max1b, v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 448, b7, ni));
              }

              float64 m0 = v_f32_reduce_max(v_f32_max_b(max0a, max0b));
              float64 m1 = v_f32_reduce_max(v_f32_max_b(max1a, max1b));
              float64 o0 = v_f32_sel_eq_f32_b(m0, ni, 0.f, m0);
              float64 o1 = v_f32_sel_eq_f32_b(m1, ni, 0.f, m1);

              TPC_EXP_NEG_F32_PAIR(a0 - o0, a1 - o0, a0, a1);
              a0 = v_f32_sel_leq_i32_b((int64)lanes, vlk0, a0, 0.f);
              a1 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 64, a1, 0.f);
              float64 sum0a = a0, sum0b = a1;
              TPC_EXP_NEG_F32_PAIR(b0 - o1, b1 - o1, b0, b1);
              b0 = v_f32_sel_leq_i32_b((int64)lanes, vlk1, b0, 0.f);
              b1 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 64, b1, 0.f);
              float64 sum1a = b0, sum1b = b1;

              if (ab > 128) {
                TPC_EXP_NEG_F32_PAIR(a2 - o0, a3 - o0, a2, a3);
                a2 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 128, a2, 0.f);
                a3 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 192, a3, 0.f);
                sum0a += a2; sum0b += a3;
                TPC_EXP_NEG_F32_PAIR(b2 - o1, b3 - o1, b2, b3);
                b2 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 128, b2, 0.f);
                b3 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 192, b3, 0.f);
                sum1a += b2; sum1b += b3;
              }
              if (ab > 256) {
                TPC_EXP_NEG_F32_PAIR(a4 - o0, a5 - o0, a4, a5);
                a4 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 256, a4, 0.f);
                a5 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 320, a5, 0.f);
                sum0a += a4; sum0b += a5;
                TPC_EXP_NEG_F32_PAIR(b4 - o1, b5 - o1, b4, b5);
                b4 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 256, b4, 0.f);
                b5 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 320, b5, 0.f);
                sum1a += b4; sum1b += b5;
              }
              if (ab > 384) {
                TPC_EXP_NEG_F32_PAIR(a6 - o0, a7 - o0, a6, a7);
                a6 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 384, a6, 0.f);
                a7 = v_f32_sel_leq_i32_b((int64)lanes, vlk0 - 448, a7, 0.f);
                sum0a += a6; sum0b += a7;
                TPC_EXP_NEG_F32_PAIR(b6 - o1, b7 - o1, b6, b7);
                b6 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 384, b6, 0.f);
                b7 = v_f32_sel_leq_i32_b((int64)lanes, vlk1 - 448, b7, 0.f);
                sum1a += b6; sum1b += b7;
              }

              float64 s0 = v_f32_reduce_add(sum0a + sum0b);
              float64 s1 = v_f32_reduce_add(sum1a + sum1b);
              const int item0 = row - first;
              acc_max = v_f32_sel_eq_u32_b(lanes, item0, m0, acc_max);
              acc_max = v_f32_sel_eq_u32_b(lanes, item0 + 1, m1, acc_max);
              acc_sum = v_f32_sel_eq_u32_b(lanes, item0, s0, acc_sum);
              acc_sum = v_f32_sel_eq_u32_b(lanes, item0 + 1, s1, acc_sum);

              float64 inv0 = tpc_rcp_nonneg_f32(s0);
              float64 inv1 = tpc_rcp_nonneg_f32(s1);
              float128 pa0 = {a0 * inv0, a1 * inv0}, pb0 = {b0 * inv1, b1 * inv1};
              q0[0] = 0; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa0, SW_LINEAR | SW_RHNE));
              q1[0] = 0; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb0, SW_LINEAR | SW_RHNE));
              if (keys > 128) {
                if (ab > 128) {
                  float128 pa1 = {a2 * inv0, a3 * inv0}, pb1 = {b2 * inv1, b3 * inv1};
                  q0[0] = 128; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa1, SW_LINEAR | SW_RHNE));
                  q1[0] = 128; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb1, SW_LINEAR | SW_RHNE));
                } else {
                  q0[0] = 128; v_bf16_st_tnsr(q0, probabilities, (bfloat128)0.f);
                  q1[0] = 128; v_bf16_st_tnsr(q1, probabilities, (bfloat128)0.f);
                }
              }
              if (keys > 256) {
                if (ab > 256) {
                  float128 pa2 = {a4 * inv0, a5 * inv0}, pb2 = {b4 * inv1, b5 * inv1};
                  q0[0] = 256; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa2, SW_LINEAR | SW_RHNE));
                  q1[0] = 256; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb2, SW_LINEAR | SW_RHNE));
                } else {
                  q0[0] = 256; v_bf16_st_tnsr(q0, probabilities, (bfloat128)0.f);
                  q1[0] = 256; v_bf16_st_tnsr(q1, probabilities, (bfloat128)0.f);
                }
              }
              if (keys > 384) {
                if (ab > 384) {
                  float128 pa3 = {a6 * inv0, a7 * inv0}, pb3 = {b6 * inv1, b7 * inv1};
                  q0[0] = 384; v_bf16_st_tnsr(q0, probabilities, convert_float128_to_bfloat128(pa3, SW_LINEAR | SW_RHNE));
                  q1[0] = 384; v_bf16_st_tnsr(q1, probabilities, convert_float128_to_bfloat128(pb3, SW_LINEAR | SW_RHNE));
                } else {
                  q0[0] = 384; v_bf16_st_tnsr(q0, probabilities, (bfloat128)0.f);
                  q1[0] = 384; v_bf16_st_tnsr(q1, probabilities, (bfloat128)0.f);
                }
              }
            }
          }
        }

        #pragma clang loop unroll(disable)
        for (; row < last; ++row) {
          int5 p = {0, row, head, kv_h, 0};
          float64 maximum = ni, sum = 0.f;
          if (row < valid_rows && max_key_blocks > 0) {
            int5 pc = {row, 0, 0, 0, 0};
            const int last_key_s = s_i32_min(s_i32_ld_g(gen_addr(pc, positions)) - key_start, valid_keys - 1);
            if (last_key_s >= 0) {
              const int active_blocks = s_i32_min(max_key_blocks, (last_key_s + 128) & ~127);
              const int active_groups = active_blocks >> 7;
              const int64 last_key = last_key_s;
              float64 max0 = ni, max1 = ni;
              float64 v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f, v4 = 0.f, v5 = 0.f, v6 = 0.f, v7 = 0.f;

              #pragma clang loop unroll(disable)
              for (int g = 0; g < active_groups; ++g) {
                const int base = g << 7;
                p[0] = base;      float64 x0 = v_f32_ld_tnsr_b(p, scores) * scale;
                p[0] = base + 64; float64 x1 = v_f32_ld_tnsr_b(p, scores) * scale;
                max0 = v_f32_max_b(max0, v_f32_sel_leq_i32_b((int64)lanes, last_key - base, x0, ni));
                max1 = v_f32_max_b(max1, v_f32_sel_leq_i32_b((int64)lanes, last_key - (base + 64), x1, ni));
                if (g == 0) { v0 = x0; v1 = x1; }
                else if (g == 1) { v2 = x0; v3 = x1; }
                else if (g == 2) { v4 = x0; v5 = x1; }
                else { v6 = x0; v7 = x1; }
              }

              maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
              float64 origin = v_f32_sel_eq_f32_b(maximum, ni, 0.f, maximum);
              float64 sum0 = 0.f, sum1 = 0.f;

              #pragma clang loop unroll(disable)
              for (int g = 0; g < active_groups; ++g) {
                const int base = g << 7;
                float64 x0 = (g < 2) ? ((g == 0) ? v0 : v2) : ((g == 2) ? v4 : v6);
                float64 x1 = (g < 2) ? ((g == 0) ? v1 : v3) : ((g == 2) ? v5 : v7);
                float64 e0, e1;
                TPC_EXP_NEG_F32_PAIR(x0 - origin, x1 - origin, e0, e1);
                e0 = v_f32_sel_leq_i32_b((int64)lanes, last_key - base, e0, 0.f);
                e1 = v_f32_sel_leq_i32_b((int64)lanes, last_key - (base + 64), e1, 0.f);
                sum0 += e0; sum1 += e1;
                if (g == 0) { v0 = e0; v1 = e1; }
                else if (g == 1) { v2 = e0; v3 = e1; }
                else if (g == 2) { v4 = e0; v5 = e1; }
                else { v6 = e0; v7 = e1; }
              }

              sum = v_f32_reduce_add(sum0 + sum1);
              float64 inverse = tpc_rcp_nonneg_f32(sum);

              #pragma clang loop unroll(disable)
              for (int g = 0; g < active_groups; ++g) {
                p[0] = g << 7;
                float64 x0 = (g < 2) ? ((g == 0) ? v0 : v2) : ((g == 2) ? v4 : v6);
                float64 x1 = (g < 2) ? ((g == 0) ? v1 : v3) : ((g == 2) ? v5 : v7);
                float128 pair = {x0 * inverse, x1 * inverse};
                v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
              }
              #pragma clang loop unroll(disable)
              for (int i = active_blocks; i < keys; i += 128) {
                p[0] = i;
                v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
              }
            } else {
              #pragma clang loop unroll(disable)
              for (int i = 0; i < keys; i += 128) {
                p[0] = i;
                v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
              }
            }
          } else {
            #pragma clang loop unroll(disable)
            for (int i = 0; i < keys; i += 128) {
              p[0] = i;
              v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
            }
          }
          const int item = row - first;
          acc_max = v_f32_sel_eq_u32_b(lanes, item, maximum, acc_max);
          acc_sum = v_f32_sel_eq_u32_b(lanes, item, sum, acc_sum);
        }

        int5 out = {first, head, kv_h, 0, 0};
        if (block_rows == 64) {
          v_f32_st_tnsr(out, tile_max, acc_max);
          v_f32_st_tnsr(out, tile_sum, acc_sum);
        } else {
          v_f32_st_tnsr_partial(out, tile_max, acc_max, (char)(block_rows - 1), 0);
          v_f32_st_tnsr_partial(out, tile_sum, acc_sum, (char)(block_rows - 1), 0);
        }
      }
    }
  }
  } else {
    const int has_softcap = (softcap > 0.0f);
    const float pre_scale = has_softcap ? (scale / softcap) : scale;
    const int max_groups = keys >> 7;

    for (int kv_h = begin[3]; kv_h < end[3]; ++kv_h) {
      for (int head = begin[0]; head < end[0]; ++head) {
        for (int block = begin[1]; block < end[1]; ++block) {
          const int first = block * block_rows;
          const int last = s_i32_min(rows, first + block_rows);
          float64 acc_max = ni, acc_sum = 0.f;

          for (int row = first; row < last; ++row) {
            int5 p = {0, row, head, kv_h, 0};
            float64 maximum = ni, sum = 0.f;
            if (row < valid_rows && max_key_blocks > 0) {
              int5 pc = {row, 0, 0, 0, 0};
              const int pos = s_i32_ld_g(gen_addr(pc, positions));
              const int first_key = (sliding_window > 0) ? s_i32_max(0, pos - sliding_window + 1 - key_start) : 0;
              const int last_key = s_i32_min(valid_keys - 1, pos - key_start);
              if (first_key <= last_key) {
                const int start_g = first_key >> 7;
                const int end_g = s_i32_min(max_groups, (last_key >> 7) + 1);
                const unsigned int win_len = (unsigned int)(last_key - first_key + 1);
                float64 max0 = ni, max1 = ni;
                float64 v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f, v4 = 0.f, v5 = 0.f, v6 = 0.f, v7 = 0.f;

                #pragma clang loop unroll(disable)
                for (int g = start_g; g < end_g; ++g) {
                  const int base = g << 7;
                  p[0] = base;      float64 x0 = v_f32_ld_tnsr_b(p, scores);
                  p[0] = base + 64; float64 x1 = v_f32_ld_tnsr_b(p, scores);
                  if (has_softcap) {
                    x0 = softcap * tpc_tanh_f32_safe(x0 * pre_scale);
                    x1 = softcap * tpc_tanh_f32_safe(x1 * pre_scale);
                  } else {
                    x0 = x0 * scale;
                    x1 = x1 * scale;
                  }
                  if (g == 0) { v0 = x0; v1 = x1; }
                  else if (g == 1) { v2 = x0; v3 = x1; }
                  else if (g == 2) { v4 = x0; v5 = x1; }
                  else { v6 = x0; v7 = x1; }
                  max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b((uint64)((int64)lanes + (base - first_key)), win_len, x0, ni));
                  max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b((uint64)((int64)lanes + (base + 64 - first_key)), win_len, x1, ni));
                }

                maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
                float64 sum0 = 0.f, sum1 = 0.f;

                #pragma clang loop unroll(disable)
                for (int g = start_g; g < end_g; ++g) {
                  const int base = g << 7;
                  float64 x0 = (g < 2) ? ((g == 0) ? v0 : v2) : ((g == 2) ? v4 : v6);
                  float64 x1 = (g < 2) ? ((g == 0) ? v1 : v3) : ((g == 2) ? v5 : v7);
                  float64 e0, e1;
                  TPC_EXP_NEG_F32_PAIR(x0 - maximum, x1 - maximum, e0, e1);
                  e0 = v_f32_sel_eq_f32_b(maximum, ni, 0.f, e0);
                  e1 = v_f32_sel_eq_f32_b(maximum, ni, 0.f, e1);
                  e0 = v_f32_sel_less_u32_b((uint64)((int64)lanes + (base - first_key)), win_len, e0, 0.f);
                  e1 = v_f32_sel_less_u32_b((uint64)((int64)lanes + (base + 64 - first_key)), win_len, e1, 0.f);
                  sum0 += e0;
                  sum1 += e1;
                  if (g == 0) { v0 = e0; v1 = e1; }
                  else if (g == 1) { v2 = e0; v3 = e1; }
                  else if (g == 2) { v4 = e0; v5 = e1; }
                  else { v6 = e0; v7 = e1; }
                }

                sum = v_f32_reduce_add(sum0 + sum1);
                float64 inverse = tpc_rcp_nonneg_f32(sum);

                for (int g = 0; g < start_g; ++g) {
                  p[0] = g << 7;
                  v_bf16_st_tnsr(p, probabilities, (bfloat128)0.f);
                }
                #pragma clang loop unroll(disable)
                for (int g = start_g; g < end_g; ++g) {
                  p[0] = g << 7;
                  float64 x0 = (g < 2) ? ((g == 0) ? v0 : v2) : ((g == 2) ? v4 : v6);
                  float64 x1 = (g < 2) ? ((g == 0) ? v1 : v3) : ((g == 2) ? v5 : v7);
                  float128 pair = {x0 * inverse, x1 * inverse};
                  v_bf16_st_tnsr(p, probabilities, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
                }
                for (int g = end_g; g < max_groups; ++g) {
                  p[0] = g << 7;
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
            const int item = row - first;
            acc_max = v_f32_sel_eq_u32_b(lanes, item, maximum, acc_max);
            acc_sum = v_f32_sel_eq_u32_b(lanes, item, sum, acc_sum);
          }

          int5 out = {first, head, kv_h, 0, 0};
          if (block_rows == 64) {
            v_f32_st_tnsr(out, tile_max, acc_max);
            v_f32_st_tnsr(out, tile_sum, acc_sum);
          } else {
            v_f32_st_tnsr_partial(out, tile_max, acc_max, (char)(block_rows - 1), 0);
            v_f32_st_tnsr_partial(out, tile_sum, acc_sum, (char)(block_rows - 1), 0);
          }
        }
      }
    }
  }
}
