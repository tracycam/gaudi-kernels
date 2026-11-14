// SPDX-License-Identifier: Apache-2.0
// Apply packed int32 token bitmask (bit b == 1 allowed, bit b == 0 -> -inf) across logits.

static inline float64 mask_64(float64 x, int base, int row, tensor bitmask,
                              int mask_words, int active_cols, int mask_tail,
                              uint64 lanes, uint64 lane_bit, float64 neg_inf) {
  if (base >= active_cols) {
    return mask_tail ? neg_inf : x;
  }
  const int word0 = base >> 5;
  const int word1 = word0 + 1;
  const unsigned int w0 = (word0 < mask_words)
      ? (unsigned int)s_i32_ld_g(gen_addr((int5){word0, row, 0, 0, 0}, bitmask))
      : 0u;
  const unsigned int w1 = (word1 < mask_words)
      ? (unsigned int)s_i32_ld_g(gen_addr((int5){word1, row, 0, 0, 0}, bitmask))
      : 0u;
  if (w0 == 0xffffffffu && w1 == 0xffffffffu && base + 64 <= active_cols) {
    return x;
  }
  const uint64 words = v_u32_sel_less_u32_b(lanes, 32u, w0, w1);
  const uint64 allowed = words & lane_bit;
  const float64 masked = v_f32_sel_eq_u32_b(allowed, 0u, neg_inf, x);
  const uint64 col_idx = lanes + (unsigned int)base;
  const float64 tail_val = mask_tail ? neg_inf : x;
  return v_f32_sel_less_u32_b(col_idx, (unsigned int)active_cols, masked, tail_val);
}

void main(tensor logits_in, tensor bitmask, tensor logits_out,
          int active_cols, int mask_tail) {
  const int width = get_dim_size(logits_in, 0);
  const int rows = get_dim_size(logits_in, 1);
  const int mask_words = get_dim_size(bitmask, 0);
#ifdef BITMASK_BF16
  const int step = 128;
#else
  const int step = 64;
#endif
  const int chunks = (width + step - 1) / step;
  const int total = rows * chunks;

  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const int per = total / ntid;
  const int rem = total % ntid;
  const int start = tid * per + s_i32_min(tid, rem);
  const int end = start + per + (tid < rem ? 1 : 0);

  const uint64 lanes = read_lane_id_4b_b();
  const uint64 lane_bit = v_u32_shl_b((uint64)1u, (int64)(lanes & 31u));
  const float64 neg_inf = (float64)(uint64)0xff800000u;

  for (int idx = start; idx < end; ++idx) {
    const int row = idx / chunks;
    const int col = (idx - row * chunks) * step;
    const char cnt = (char)(s_i32_min(step, width - col) - 1);
    int5 p = {col, row, 0, 0, 0};
#ifdef BITMASK_BF16
    const float128 v = convert_bfloat128_to_float128(
        v_bf16_ld_tnsr_partial_b(p, logits_in, cnt, 0), SW_LINEAR);
    float64 o0 = mask_64(v.v1, col, row, bitmask, mask_words, active_cols,
                         mask_tail, lanes, lane_bit, neg_inf);
    float64 o1 = mask_64(v.v2, col + 64, row, bitmask, mask_words, active_cols,
                         mask_tail, lanes, lane_bit, neg_inf);
    float128 out = {o0, o1};
    v_bf16_st_tnsr_partial(p, logits_out,
                           convert_float128_to_bfloat128(out, SW_LINEAR | SW_RHNE),
                           cnt, 0);
#else
    const float64 x = v_f32_ld_tnsr_partial_b(p, logits_in, cnt, 0);
    const float64 out = mask_64(x, col, row, bitmask, mask_words, active_cols,
                                mask_tail, lanes, lane_bit, neg_inf);
    v_f32_st_tnsr_partial(p, logits_out, out, cnt, 0);
#endif
  }
}
