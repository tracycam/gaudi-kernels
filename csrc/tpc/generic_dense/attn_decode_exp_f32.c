// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Long single-row Decode (512 < keys <= 8192). Exponentials remain an internal
// graph tensor; base-2 exp2f avoids Cody-Waite range reduction and skips masked
// trailing key blocks.
void main(tensor scores, tensor positions, tensor header, tensor exponent, tensor sums,
          float scale, float softcap, int sliding_window) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 z = {0, 0, 0, 0, 0}, h1 = {1, 0, 0, 0, 0}, h2 = {2, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(z, header));
  const int valid_keys = s_i32_ld_g(gen_addr(h1, header));
  const int key_start = s_i32_ld_g(gen_addr(h2, header));
  const int position = s_i32_ld_g(gen_addr(z, positions));
  const int keys = get_dim_size(scores, 0);
  const uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;
  const int plain_causal = (softcap <= 0.0f && sliding_window <= 0);

  if (plain_causal) {
  const int visible = valid_rows ? s_i32_max(0, s_i32_min(valid_keys, position - key_start + 1)) : 0;
  const int full256 = visible & ~255;
  const int active128 = s_i32_min(keys, (visible + 127) & ~127);

  for (int head = begin[0]; head < end[0]; ++head) {
    float64 max0 = ni, max1 = ni;
    int i = 0;
    #pragma clang loop unroll(disable)
    for (; i < full256; i += 256) {
      int5 p = {i, 0, head, 0, 0};
      float64 a0 = v_f32_ld_tnsr_b(p, scores) * scale; p[0] = i + 64;
      float64 a1 = v_f32_ld_tnsr_b(p, scores) * scale; p[0] = i + 128;
      float64 a2 = v_f32_ld_tnsr_b(p, scores) * scale; p[0] = i + 192;
      float64 a3 = v_f32_ld_tnsr_b(p, scores) * scale;
      max0 = v_f32_max_b(max0, v_f32_max_b(a0, a2));
      max1 = v_f32_max_b(max1, v_f32_max_b(a1, a3));
    }
    #pragma clang loop unroll(disable)
    for (; i < active128; i += 128) {
      int5 p = {i, 0, head, 0, 0};
      float64 a = v_f32_ld_tnsr_b(p, scores) * scale; p[0] = i + 64;
      float64 b = v_f32_ld_tnsr_b(p, scores) * scale;
      max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64, visible - i)), a, ni));
      max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64, visible - i - 64)), b, ni));
    }
    float64 maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));
    float64 origin = v_f32_sel_eq_f32_b(maximum, ni, 0.f, maximum);
    float64 sum0 = 0.f, sum1 = 0.f;

    i = 0;
    #pragma clang loop unroll(disable)
    for (; i < full256; i += 256) {
      int5 p = {i, 0, head, 0, 0};
      float64 a0 = v_f32_ld_tnsr_b(p, scores) * scale - origin; p[0] = i + 64;
      float64 a1 = v_f32_ld_tnsr_b(p, scores) * scale - origin; p[0] = i + 128;
      float64 a2 = v_f32_ld_tnsr_b(p, scores) * scale - origin; p[0] = i + 192;
      float64 a3 = v_f32_ld_tnsr_b(p, scores) * scale - origin;
      float64 e0, e1, e2, e3;
      TPC_EXP_NEG_F32_QUAD(a0, a1, a2, a3, e0, e1, e2, e3);
      sum0 += e0 + e2;
      sum1 += e1 + e3;
      p[0] = i;       v_f32_st_tnsr(p, exponent, e0);
      p[0] = i + 64;  v_f32_st_tnsr(p, exponent, e1);
      p[0] = i + 128; v_f32_st_tnsr(p, exponent, e2);
      p[0] = i + 192; v_f32_st_tnsr(p, exponent, e3);
    }
    #pragma clang loop unroll(disable)
    for (; i < active128; i += 128) {
      int5 p = {i, 0, head, 0, 0};
      float64 a0 = v_f32_ld_tnsr_b(p, scores) * scale - origin; p[0] = i + 64;
      float64 a1 = v_f32_ld_tnsr_b(p, scores) * scale - origin;
      float64 e0, e1;
      TPC_EXP_NEG_F32_PAIR(a0, a1, e0, e1);
      e0 = v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64, visible - i)), e0, 0.f);
      e1 = v_f32_sel_less_u32_b(lanes, s_i32_max(0, s_i32_min(64, visible - i - 64)), e1, 0.f);
      sum0 += e0;
      sum1 += e1;
      p[0] = i;      v_f32_st_tnsr(p, exponent, e0);
      p[0] = i + 64; v_f32_st_tnsr(p, exponent, e1);
    }
    #pragma clang loop unroll(disable)
    for (; i < keys; i += 128) {
      int5 p = {i, 0, head, 0, 0};
      v_f32_st_tnsr(p, exponent, 0.f);
      p[0] = i + 64;
      v_f32_st_tnsr(p, exponent, 0.f);
    }
    float64 sum = v_f32_reduce_add(sum0 + sum1);
    int5 stat = {0, head, 0, 0, 0};
    v_f32_st_tnsr(stat, sums, sum);
  }
  } else {
    const int has_softcap = (softcap > 0.0f);
    const float pre_scale = has_softcap ? (scale / softcap) : scale;
    const int first_key = (sliding_window > 0) ? s_i32_max(0, position - sliding_window + 1 - key_start) : 0;
    const int last_key = valid_rows ? s_i32_min(valid_keys - 1, position - key_start) : -1;

    if (first_key > last_key) {
      for (int head = begin[0]; head < end[0]; ++head) {
        for (int i = 0; i < keys; i += 128) {
          int5 p = {i, 0, head, 0, 0};
          v_f32_st_tnsr(p, exponent, 0.f);
          p[0] = i + 64;
          v_f32_st_tnsr(p, exponent, 0.f);
        }
        int5 stat = {0, head, 0, 0, 0};
        v_f32_st_tnsr(stat, sums, (float64)0.f);
      }
    } else {
      const int start128 = first_key & ~127;
      const int end128 = s_i32_min(keys, (last_key + 128) & ~127);
      const unsigned int win_len = (unsigned int)(last_key - first_key + 1);

      for (int head = begin[0]; head < end[0]; ++head) {
        float64 max0 = ni, max1 = ni;
        #pragma clang loop unroll(disable)
        for (int i = start128; i < end128; i += 128) {
          int5 p = {i, 0, head, 0, 0};
          float64 a = v_f32_ld_tnsr_b(p, scores);
          p[0] = i + 64;
          float64 b = v_f32_ld_tnsr_b(p, scores);
          if (has_softcap) {
            a = softcap * tpc_tanh_f32_safe(a * pre_scale);
            b = softcap * tpc_tanh_f32_safe(b * pre_scale);
          } else {
            a = a * scale;
            b = b * scale;
          }
          max0 = v_f32_max_b(max0, v_f32_sel_less_u32_b((uint64)((int64)lanes + (i - first_key)), win_len, a, ni));
          max1 = v_f32_max_b(max1, v_f32_sel_less_u32_b((uint64)((int64)lanes + (i + 64 - first_key)), win_len, b, ni));
        }
        float64 maximum = v_f32_reduce_max(v_f32_max_b(max0, max1));

        for (int i = 0; i < start128; i += 128) {
          int5 p = {i, 0, head, 0, 0};
          v_f32_st_tnsr(p, exponent, 0.f);
          p[0] = i + 64;
          v_f32_st_tnsr(p, exponent, 0.f);
        }

        float64 sum0 = 0.f, sum1 = 0.f;
        #pragma clang loop unroll(disable)
        for (int i = start128; i < end128; i += 128) {
          int5 p = {i, 0, head, 0, 0};
          float64 a = v_f32_ld_tnsr_b(p, scores);
          p[0] = i + 64;
          float64 b = v_f32_ld_tnsr_b(p, scores);
          if (has_softcap) {
            a = softcap * tpc_tanh_f32_safe(a * pre_scale);
            b = softcap * tpc_tanh_f32_safe(b * pre_scale);
          } else {
            a = a * scale;
            b = b * scale;
          }
          float64 e0, e1;
          TPC_EXP_NEG_F32_PAIR(a - maximum, b - maximum, e0, e1);
          e0 = v_f32_sel_eq_f32_b(maximum, ni, 0.f, e0);
          e1 = v_f32_sel_eq_f32_b(maximum, ni, 0.f, e1);
          e0 = v_f32_sel_less_u32_b((uint64)((int64)lanes + (i - first_key)), win_len, e0, 0.f);
          e1 = v_f32_sel_less_u32_b((uint64)((int64)lanes + (i + 64 - first_key)), win_len, e1, 0.f);
          p[0] = i;
          v_f32_st_tnsr(p, exponent, e0);
          p[0] = i + 64;
          v_f32_st_tnsr(p, exponent, e1);
          sum0 += e0;
          sum1 += e1;
        }

        for (int i = end128; i < keys; i += 128) {
          int5 p = {i, 0, head, 0, 0};
          v_f32_st_tnsr(p, exponent, 0.f);
          p[0] = i + 64;
          v_f32_st_tnsr(p, exponent, 0.f);
        }
        float64 sum = v_f32_reduce_add(sum0 + sum1);
        int5 stat = {0, head, 0, 0, 0};
        v_f32_st_tnsr(stat, sums, sum);
      }
    }
  }
}
