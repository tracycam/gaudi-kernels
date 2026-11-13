// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Maps any non-NaN IEEE-754 FP32 value to a strictly monotonic signed int32
// across [-Inf, +Inf] with -0.0f == +0.0f == 0.
static inline int64 float_to_ord_i32(float64 v) {
  uint64 b = (uint64)v;
  b = v_u32_sel_eq_u32_b(b, 0x80000000u, 0u, b);
  int64 neg = (int64)(b ^ 0x7fffffffu);
  int64 pos = (int64)b;
  return v_i32_sel_geq_u32_b(b, 0x80000000u, neg, pos);
}

#define MULHILO32(A, B, LO, HI) \
  do { \
    const unsigned int _a = (A); \
    const unsigned int _b = (B); \
    const unsigned int _a0 = _a & 0xffffu; \
    const unsigned int _a1 = _a >> 16; \
    const unsigned int _b0 = _b & 0xffffu; \
    const unsigned int _b1 = _b >> 16; \
    const unsigned int _p00 = _a0 * _b0; \
    const unsigned int _p01 = _a0 * _b1; \
    const unsigned int _p10 = _a1 * _b0; \
    const unsigned int _p11 = _a1 * _b1; \
    const unsigned int _mid = (_p00 >> 16) + (_p01 & 0xffffu) + (_p10 & 0xffffu); \
    (LO) = (_p00 & 0xffffu) | ((_mid & 0xffffu) << 16); \
    (HI) = _p11 + (_p01 >> 16) + (_p10 >> 16) + (_mid >> 16); \
  } while (0)

static inline float philox_uniform_f32(unsigned int c0, unsigned int k0, unsigned int k1) {
  unsigned int c1 = 0u, c2 = 0u, c3 = 0u;
  for (int r = 0; r < 10; ++r) {
    unsigned int lo0, hi0, lo1, hi1;
    MULHILO32(0xD2511F53u, c0, lo0, hi0);
    MULHILO32(0xCD9E8D57u, c2, lo1, hi1);
    const unsigned int nc0 = hi1 ^ c1 ^ k0;
    const unsigned int nc1 = lo1;
    const unsigned int nc2 = hi0 ^ c3 ^ k1;
    const unsigned int nc3 = lo0;
    c0 = nc0;
    c1 = nc1;
    c2 = nc2;
    c3 = nc3;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return (float)(c0 >> 8) * (1.0f / 16777216.0f);
}

#ifdef SAMPLE_BF16
#define LOAD_LOGITS_PAIR(BASE, ROW, VOCAB, NI, V0, V1) \
  do { \
    const int _rem = s_i32_min(128, (VOCAB) - (BASE)); \
    int5 _p = {(BASE), (ROW), 0, 0, 0}; \
    float128 _pair = convert_bfloat128_to_float128( \
        v_bf16_ld_tnsr_partial_b(_p, logits, (char)(_rem - 1), 0), SW_LINEAR); \
    (V0) = _pair.v1; \
    (V1) = _pair.v2; \
  } while (0)
#else
#define LOAD_LOGITS_PAIR(BASE, ROW, VOCAB, NI, V0, V1) \
  do { \
    const int _rem0 = s_i32_min(64, (VOCAB) - (BASE)); \
    int5 _p0 = {(BASE), (ROW), 0, 0, 0}; \
    (V0) = v_f32_ld_tnsr_partial_b(_p0, logits, (char)(_rem0 - 1), 0); \
    if ((VOCAB) - (BASE) > 64) { \
      const int _rem1 = s_i32_min(64, (VOCAB) - (BASE) - 64); \
      int5 _p1 = {(BASE) + 64, (ROW), 0, 0, 0}; \
      (V1) = v_f32_ld_tnsr_partial_b(_p1, logits, (char)(_rem1 - 1), 0); \
    } else { \
      (V1) = (NI); \
    } \
  } while (0)
#endif

#define SANITIZE_PAIR(BASE, VOCAB, NAN_FLAG, NI, LANES, V0, V1, VAL0, VAL1) \
  do { \
    const int _val0 = s_i32_max(0, s_i32_min(64, (VOCAB) - (BASE))); \
    const int _val1 = s_i32_max(0, s_i32_min(64, (VOCAB) - ((BASE) + 64))); \
    (VAL0) = (unsigned int)_val0; \
    (VAL1) = (unsigned int)_val1; \
    if ((NAN_FLAG) != 0) { \
      (V0) = v_f32_sel_grt_u32_b((uint64)(V0) & 0x7fffffffu, 0x7f800000u, (NI), (V0)); \
      (V1) = v_f32_sel_grt_u32_b((uint64)(V1) & 0x7fffffffu, 0x7f800000u, (NI), (V1)); \
    } \
    (V0) = v_f32_sel_less_u32_b((LANES), (VAL0), (V0), (NI)); \
    (V1) = v_f32_sel_less_u32_b((LANES), (VAL1), (V1), (NI)); \
  } while (0)

void main(tensor logits, tensor temperatures, tensor top_ps, tensor top_ks,
          tensor min_ps, tensor seeds, tensor output_ids,
          int vocab_size, int nan_to_neg_inf) {
  const int rows = get_dim_size(logits, 1);
  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const uint64 lanes = read_lane_id_4b_b();
  const float ni = -1.0f / 0.0f;

  for (int row = tid; row < rows; row += ntid) {
    int5 rpos = {row, 0, 0, 0, 0};
    int5 out_pos = {row, 0, 0, 0, 0};
    if (vocab_size <= 0) {
      int64 z = 0;
      v_i32_st_tnsr_partial(out_pos, output_ids, z, 0, 0);
      continue;
    }

    const float temp = s_f32_ld_g(gen_addr(rpos, temperatures));
    const float top_p = s_f32_ld_g(gen_addr(rpos, top_ps));
    const int top_k_raw = s_i32_ld_g(gen_addr(rpos, top_ks));
    const float min_p = s_f32_ld_g(gen_addr(rpos, min_ps));
    int5 spos0 = {0, row, 0, 0, 0};
    int5 spos1 = {1, row, 0, 0, 0};
    const unsigned int seed0 = (unsigned int)s_i32_ld_g(gen_addr(spos0, seeds));
    const unsigned int seed1 = (unsigned int)s_i32_ld_g(gen_addr(spos1, seeds));
    const int eff_top_k = (top_k_raw <= 0 || top_k_raw > vocab_size) ? vocab_size : top_k_raw;

    // Pass 1: max logit, lowest-index argmax, and [min_ord, max_ord] for Top-K.
    float64 best_val = ni;
    int64 best_idx = 2147483647;
    int64 first_nan = 2147483647;
    int64 min_ord = 2147483647;
    int64 max_ord = -2147483647 - 1;

    for (int base = 0; base < vocab_size; base += 128) {
      float64 v0, v1;
      LOAD_LOGITS_PAIR(base, row, vocab_size, ni, v0, v1);
      unsigned int val0, val1;
      SANITIZE_PAIR(base, vocab_size, nan_to_neg_inf, ni, lanes, v0, v1, val0, val1);

      int64 idx0 = v_i32_sel_less_u32_b(lanes, val0, (int64)lanes + base, 2147483647);
      int64 idx1 = v_i32_sel_less_u32_b(lanes, val1, (int64)lanes + (base + 64), 2147483647);

      if (nan_to_neg_inf == 0) {
        int64 nan0 = v_i32_sel_grt_u32_b((uint64)v0 & 0x7fffffffu, 0x7f800000u, idx0, 2147483647);
        int64 nan1 = v_i32_sel_grt_u32_b((uint64)v1 & 0x7fffffffu, 0x7f800000u, idx1, 2147483647);
        first_nan = v_i32_min_b(first_nan, v_i32_min_b(nan0, nan1));
      }

      int64 b_idx = v_i32_sel_grt_f32_b(
          v0, best_val, idx0,
          v_i32_sel_eq_f32_b(v0, best_val, v_i32_min_b(best_idx, idx0), best_idx));
      float64 b_val = v_f32_max_b(best_val, v0);

      best_idx = v_i32_sel_grt_f32_b(
          v1, b_val, idx1,
          v_i32_sel_eq_f32_b(v1, b_val, v_i32_min_b(b_idx, idx1), b_idx));
      best_val = v_f32_max_b(b_val, v1);

      int64 o0 = float_to_ord_i32(v0);
      int64 o1 = float_to_ord_i32(v1);
      min_ord = v_i32_min_b(min_ord, v_i32_sel_less_u32_b(lanes, val0, o0, 2147483647));
      min_ord = v_i32_min_b(min_ord, v_i32_sel_less_u32_b(lanes, val1, o1, 2147483647));
      max_ord = v_i32_max_b(max_ord, v_i32_sel_less_u32_b(lanes, val0, o0, -2147483647 - 1));
      max_ord = v_i32_max_b(max_ord, v_i32_sel_less_u32_b(lanes, val1, o1, -2147483647 - 1));
    }

    float64 max_vec = v_f32_reduce_max(best_val);
    int64 argmax_vec = v_i32_reduce_min(v_i32_sel_eq_f32_b(best_val, max_vec, best_idx, 2147483647));
    if (nan_to_neg_inf == 0) {
      int64 nan_idx = v_i32_reduce_min(first_nan);
      argmax_vec = v_i32_sel_less_i32_b(nan_idx, vocab_size, nan_idx, argmax_vec);
    }

    if (temp <= 0.0f || eff_top_k == 1) {
      v_i32_st_tnsr_partial(out_pos, output_ids, argmax_vec, 0, 0);
      continue;
    }

    // Pass 2: Exact Top-K threshold + tie quota via 16-pass 3-pivot search
    // over unsigned-biased order keys u = (uint32)(ord ^ 0x80000000u).
    int64 thresh_ord = -2147483647 - 1;
    int64 cutoff_tie_idx = vocab_size;
    if (eff_top_k < vocab_size) {
      uint64 lo_u = (uint64)(v_i32_reduce_min(min_ord)) ^ 0x80000000u;
      uint64 hi_u = (uint64)(v_i32_reduce_max(max_ord)) ^ 0x80000000u;

      for (int step = 0; step < 18; ++step) {
        uint64 span = hi_u - lo_u;
        uint64 q = span >> 2;
        uint64 m1_u = v_u32_min_b(hi_u, lo_u + v_u32_max_b(q, 1u));
        uint64 m2_u = v_u32_min_b(hi_u, lo_u + v_u32_max_b(span >> 1, 2u));
        uint64 m3_u = v_u32_sel_less_u32_b(span, 4u, hi_u, hi_u - q);
        int64 c1 = 0, c2 = 0, c3 = 0;

        for (int base = 0; base < vocab_size; base += 128) {
          float64 v0, v1;
          LOAD_LOGITS_PAIR(base, row, vocab_size, ni, v0, v1);
          unsigned int val0, val1;
          SANITIZE_PAIR(base, vocab_size, nan_to_neg_inf, ni, lanes, v0, v1, val0, val1);
          uint64 o0 = (uint64)float_to_ord_i32(v0) ^ 0x80000000u;
          uint64 o1 = (uint64)float_to_ord_i32(v1) ^ 0x80000000u;
          c1 += v_i32_sel_less_u32_b(lanes, val0, v_i32_sel_geq_u32_b(o0, m1_u, 1, 0), 0);
          c1 += v_i32_sel_less_u32_b(lanes, val1, v_i32_sel_geq_u32_b(o1, m1_u, 1, 0), 0);
          c2 += v_i32_sel_less_u32_b(lanes, val0, v_i32_sel_geq_u32_b(o0, m2_u, 1, 0), 0);
          c2 += v_i32_sel_less_u32_b(lanes, val1, v_i32_sel_geq_u32_b(o1, m2_u, 1, 0), 0);
          c3 += v_i32_sel_less_u32_b(lanes, val0, v_i32_sel_geq_u32_b(o0, m3_u, 1, 0), 0);
          c3 += v_i32_sel_less_u32_b(lanes, val1, v_i32_sel_geq_u32_b(o1, m3_u, 1, 0), 0);
        }
        int64 t1 = v_i32_reduce_add(c1);
        int64 t2 = v_i32_reduce_add(c2);
        int64 t3 = v_i32_reduce_add(c3);

        uint64 nlo = lo_u, nhi = hi_u;
        nhi = v_u32_sel_less_i32_b(t1, eff_top_k, v_u32_max_b(m1_u, lo_u + 1u) - 1u,
                                   v_u32_sel_less_i32_b(t2, eff_top_k, m2_u - 1u,
                                                        v_u32_sel_less_i32_b(t3, eff_top_k, m3_u - 1u, hi_u)));
        nlo = v_u32_sel_geq_i32_b(t3, eff_top_k, m3_u,
                                  v_u32_sel_geq_i32_b(t2, eff_top_k, m2_u,
                                                      v_u32_sel_geq_i32_b(t1, eff_top_k, m1_u, lo_u)));
        lo_u = v_u32_sel_eq_u32_b(span, 0u, lo_u, nlo);
        hi_u = v_u32_sel_eq_u32_b(span, 0u, hi_u, v_u32_max_b(nlo, nhi));
      }
      thresh_ord = (int64)(lo_u ^ 0x80000000u);

      int64 gt_cnt_acc = 0;
      for (int base = 0; base < vocab_size; base += 128) {
        float64 v0, v1;
        LOAD_LOGITS_PAIR(base, row, vocab_size, ni, v0, v1);
        unsigned int val0, val1;
        SANITIZE_PAIR(base, vocab_size, nan_to_neg_inf, ni, lanes, v0, v1, val0, val1);
        int64 o0 = float_to_ord_i32(v0);
        int64 o1 = float_to_ord_i32(v1);
        gt_cnt_acc += v_i32_sel_less_u32_b(lanes, val0, v_i32_sel_grt_i32_b(o0, thresh_ord, 1, 0), 0);
        gt_cnt_acc += v_i32_sel_less_u32_b(lanes, val1, v_i32_sel_grt_i32_b(o1, thresh_ord, 1, 0), 0);
      }
      int64 tie_quota = (int64)eff_top_k - v_i32_reduce_add(gt_cnt_acc);

      int64 running_ties = 0, tie_found = 0, tie_base = 0, tie_prev = 0, tie_vec = 0;
      for (int base = 0; base < vocab_size; base += 128) {
        float64 v0, v1;
        LOAD_LOGITS_PAIR(base, row, vocab_size, ni, v0, v1);
        unsigned int val0, val1;
        SANITIZE_PAIR(base, vocab_size, nan_to_neg_inf, ni, lanes, v0, v1, val0, val1);
        int64 o0 = float_to_ord_i32(v0);
        int64 o1 = float_to_ord_i32(v1);

        int64 m0 = v_i32_sel_less_u32_b(lanes, val0, v_i32_sel_eq_i32_b(o0, thresh_ord, 1, 0), 0);
        int64 next0 = running_ties + v_i32_reduce_add(m0);
        int64 hit0 = v_i32_sel_eq_i32_b(tie_found, 0, v_i32_sel_geq_i32_b(next0, tie_quota, 1, 0), 0);
        tie_base = v_i32_sel_eq_i32_b(hit0, 1, base, tie_base);
        tie_prev = v_i32_sel_eq_i32_b(hit0, 1, running_ties, tie_prev);
        tie_vec  = v_i32_sel_eq_i32_b(hit0, 1, m0, tie_vec);
        tie_found = v_i32_sel_eq_i32_b(hit0, 1, 1, tie_found);
        running_ties = next0;

        if (val1 > 0u) {
          int64 m1 = v_i32_sel_less_u32_b(lanes, val1, v_i32_sel_eq_i32_b(o1, thresh_ord, 1, 0), 0);
          int64 next1 = running_ties + v_i32_reduce_add(m1);
          int64 hit1 = v_i32_sel_eq_i32_b(tie_found, 0, v_i32_sel_geq_i32_b(next1, tie_quota, 1, 0), 0);
          tie_base = v_i32_sel_eq_i32_b(hit1, 1, base + 64, tie_base);
          tie_prev = v_i32_sel_eq_i32_b(hit1, 1, running_ties, tie_prev);
          tie_vec  = v_i32_sel_eq_i32_b(hit1, 1, m1, tie_vec);
          tie_found = v_i32_sel_eq_i32_b(hit1, 1, 1, tie_found);
          running_ties = next1;
        }
      }
      int64 l0 = 0, l1 = 63;
      for (int b = 0; b < 6; ++b) {
        int64 mid = (l0 + l1) >> 1;
        int64 pref = v_i32_reduce_add(v_i32_sel_leq_i32_b((int64)lanes, mid, tie_vec, 0));
        int64 cum = tie_prev + pref;
        l1 = v_i32_sel_geq_i32_b(cum, tie_quota, mid, l1);
        l0 = v_i32_sel_less_i32_b(cum, tie_quota, mid + 1, l0);
      }
      cutoff_tie_idx = v_i32_sel_eq_i32_b(tie_found, 1, tie_base + l0, vocab_size);
    }

    const float inv_temp = 1.0f / temp;
    const int do_topk = (eff_top_k < vocab_size);
    const int do_minp = (min_p > 0.0f);

    #define COMPUTE_WEIGHTS_PAIR(BASE, W0, W1, VAL0, VAL1) \
      do { \
        float64 _v0, _v1; \
        LOAD_LOGITS_PAIR((BASE), row, vocab_size, ni, _v0, _v1); \
        SANITIZE_PAIR((BASE), vocab_size, nan_to_neg_inf, ni, lanes, _v0, _v1, (VAL0), (VAL1)); \
        float64 _w0 = tpc_exp_f32_safe((_v0 - max_vec) * inv_temp); \
        float64 _w1 = tpc_exp_f32_safe((_v1 - max_vec) * inv_temp); \
        _w0 = v_f32_sel_eq_f32_b(max_vec, ni, 0.0f, v_f32_sel_eq_f32_b(_v0, ni, 0.0f, _w0)); \
        _w1 = v_f32_sel_eq_f32_b(max_vec, ni, 0.0f, v_f32_sel_eq_f32_b(_v1, ni, 0.0f, _w1)); \
        _w0 = v_f32_sel_less_u32_b(lanes, (VAL0), _w0, 0.0f); \
        _w1 = v_f32_sel_less_u32_b(lanes, (VAL1), _w1, 0.0f); \
        if (do_topk) { \
          int64 _o0 = float_to_ord_i32(_v0); \
          int64 _o1 = float_to_ord_i32(_v1); \
          int64 _i0 = (int64)lanes + (BASE); \
          int64 _i1 = (int64)lanes + ((BASE) + 64); \
          int64 _keep0 = v_i32_sel_grt_i32_b(_o0, thresh_ord, 1, \
                            v_i32_sel_eq_i32_b(_o0, thresh_ord, v_i32_sel_leq_i32_b(_i0, cutoff_tie_idx, 1, 0), 0)); \
          int64 _keep1 = v_i32_sel_grt_i32_b(_o1, thresh_ord, 1, \
                            v_i32_sel_eq_i32_b(_o1, thresh_ord, v_i32_sel_leq_i32_b(_i1, cutoff_tie_idx, 1, 0), 0)); \
          _w0 = v_f32_sel_eq_i32_b(_keep0, 1, _w0, 0.0f); \
          _w1 = v_f32_sel_eq_i32_b(_keep1, 1, _w1, 0.0f); \
        } \
        if (do_minp) { \
          _w0 = v_f32_sel_geq_f32_b(_w0, min_p, _w0, 0.0f); \
          _w1 = v_f32_sel_geq_f32_b(_w1, min_p, _w1, 0.0f); \
        } \
        (W0) = _w0; \
        (W1) = _w1; \
      } while (0)

    // Pass 3: compute total unnormalized weight after Top-K & min_p.
    float64 sum_acc = 0.0f;
    for (int base = 0; base < vocab_size; base += 128) {
      float64 w0, w1;
      unsigned int val0, val1;
      COMPUTE_WEIGHTS_PAIR(base, w0, w1, val0, val1);
      sum_acc += w0 + w1;
    }
    float64 total_w = v_f32_reduce_add(sum_acc);

    // Pass 4: Top-P mass threshold (24-step binary search on [0.0f, 1.0f]).
    float64 tau_p = 0.0f;
    float64 kept_w = total_w;
    if (top_p <= 0.0f || top_p < 1.0f) {
      if (top_p <= 0.0f) {
        tau_p = 1.0f;
      } else {
        float64 target_mass = total_w * top_p;
        float64 lo_p = 0.0f, hi_p = 1.0f;
        for (int step = 0; step < 24; ++step) {
          float64 mid_p = 0.5f * (lo_p + hi_p);
          float64 m_acc = 0.0f;
          for (int base = 0; base < vocab_size; base += 128) {
            float64 w0, w1;
            unsigned int val0, val1;
            COMPUTE_WEIGHTS_PAIR(base, w0, w1, val0, val1);
            m_acc += v_f32_sel_geq_f32_b(w0, mid_p, w0, 0.0f) + v_f32_sel_geq_f32_b(w1, mid_p, w1, 0.0f);
          }
          float64 m_tot = v_f32_reduce_add(m_acc);
          lo_p = v_f32_sel_geq_f32_b(m_tot, target_mass, mid_p, lo_p);
          hi_p = v_f32_sel_less_f32_b(m_tot, target_mass, mid_p, hi_p);
        }
        tau_p = lo_p;
      }
      float64 k_acc = 0.0f;
      for (int base = 0; base < vocab_size; base += 128) {
        float64 w0, w1;
        unsigned int val0, val1;
        COMPUTE_WEIGHTS_PAIR(base, w0, w1, val0, val1);
        k_acc += v_f32_sel_geq_f32_b(w0, tau_p, w0, 0.0f) + v_f32_sel_geq_f32_b(w1, tau_p, w1, 0.0f);
      }
      kept_w = v_f32_reduce_add(k_acc);
    }

    // Pass 5: Deterministic Philox-4x32-10 CDF scan (batch-position invariant).
    const float u = philox_uniform_f32(0u, seed0, seed1);
    float64 target_u = kept_w * u;
    float64 running_sum = 0.0f;
    int64 found = 0, win_base = 0;
    float64 win_prev = 0.0f, win_w = 0.0f;
    int64 last_nz = -1;

    for (int base = 0; base < vocab_size; base += 128) {
      float64 w0, w1;
      unsigned int val0, val1;
      COMPUTE_WEIGHTS_PAIR(base, w0, w1, val0, val1);
      w0 = v_f32_sel_geq_f32_b(w0, tau_p, w0, 0.0f);
      w1 = v_f32_sel_geq_f32_b(w1, tau_p, w1, 0.0f);

      last_nz = v_i32_max_b(last_nz, v_i32_reduce_max(v_i32_sel_grt_f32_b(w0, 0.0f, (int64)lanes + base, -1)));
      float64 next0 = running_sum + v_f32_reduce_add(w0);
      int64 hit0 = v_i32_sel_eq_i32_b(found, 0, v_i32_sel_grt_f32_b(next0, target_u, 1, 0), 0);
      win_base = v_i32_sel_eq_i32_b(hit0, 1, base, win_base);
      win_prev = v_f32_sel_eq_i32_b(hit0, 1, running_sum, win_prev);
      win_w = v_f32_sel_eq_i32_b(hit0, 1, w0, win_w);
      found = v_i32_sel_eq_i32_b(hit0, 1, 1, found);
      running_sum = next0;

      if (val1 > 0u) {
        last_nz = v_i32_max_b(last_nz, v_i32_reduce_max(v_i32_sel_grt_f32_b(w1, 0.0f, (int64)lanes + (base + 64), -1)));
        float64 next1 = running_sum + v_f32_reduce_add(w1);
        int64 hit1 = v_i32_sel_eq_i32_b(found, 0, v_i32_sel_grt_f32_b(next1, target_u, 1, 0), 0);
        win_base = v_i32_sel_eq_i32_b(hit1, 1, base + 64, win_base);
        win_prev = v_f32_sel_eq_i32_b(hit1, 1, running_sum, win_prev);
        win_w = v_f32_sel_eq_i32_b(hit1, 1, w1, win_w);
        found = v_i32_sel_eq_i32_b(hit1, 1, 1, found);
        running_sum = next1;
      }
    }

    int64 l0 = 0, l1 = 63;
    for (int b = 0; b < 6; ++b) {
      int64 mid = (l0 + l1) >> 1;
      float64 pref = v_f32_reduce_add(v_f32_sel_leq_i32_b((int64)lanes, mid, win_w, 0.0f));
      float64 cum = win_prev + pref;
      l1 = v_i32_sel_grt_f32_b(cum, target_u, mid, l1);
      l0 = v_i32_sel_leq_f32_b(cum, target_u, mid + 1, l0);
    }
    int64 fallback_id = v_i32_sel_geq_i32_b(last_nz, 0, last_nz, argmax_vec);
    int64 sampled_id = v_i32_sel_eq_i32_b(found, 1, win_base + l0, fallback_id);
    v_i32_st_tnsr_partial(out_pos, output_ids, sampled_id, 0, 0);

    #undef COMPUTE_WEIGHTS_PAIR
  }
}
