// SPDX-License-Identifier: Apache-2.0
// Refine BF16 dot products whose MME reductions may cross a rounding boundary.
static inline float128 add_pair(float64 a, float64 al, float64 b, float64 bl) {
  float64 sum = a + b;
  float64 z = sum - a;
  float64 roundoff = (a - (sum - z)) + (b - z);
  float128 result = {sum, al + bl + roundoff};
  return result;
}

static inline float128 reduce_pair(float64 high, float64 low) {
  const int flags = MkWrA(3, 3, 3, 3);
  float64 other_high = v_f32_mov_dual_group_all_b(high, 0xffffffff, 1, 0, 3, 2, flags, 0);
  float64 other_low = v_f32_mov_dual_group_all_b(low, 0xffffffff, 1, 0, 3, 2, flags, 0);
  float128 pair = add_pair(high, low, other_high, other_low);
  other_high = v_f32_mov_dual_group_all_b(pair.v1, 0xffffffff, 2, 3, 0, 1, flags, 0);
  other_low = v_f32_mov_dual_group_all_b(pair.v2, 0xffffffff, 2, 3, 0, 1, flags, 0);
  pair = add_pair(pair.v1, pair.v2, other_high, other_low);
  const int groups = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                     SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;
  other_high = v_f32_mov_group_b(pair.v1, -1, groups, pair.v1);
  other_low = v_f32_mov_group_b(pair.v2, -1, groups, pair.v2);
  pair = add_pair(pair.v1, pair.v2, other_high, other_low);
  uint64 lanes = read_lane_id_4b_b();
  for (int shift = 4; shift >= 1; shift >>= 1) {
    uchar256 direction = (uchar256)((128 + ((lanes & 7) ^ shift) + ((lanes & 8) << 2)) * 0x01010101);
    other_high = v_f32_shuffle_b(pair.v1, direction, 0, 0);
    other_low = v_f32_shuffle_b(pair.v2, direction, 0, 0);
    pair = add_pair(pair.v1, pair.v2, other_high, other_low);
  }
  return pair;
}

void main(tensor input, tensor weight, tensor first, tensor second,
          tensor full_f32, tensor split_f32, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(input, 0);
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int column = begin[0]; column < end[0]; ++column) {
      int5 out = {column, row, 0, 0, 0};
      const bf16 left = s_bf16_ld_g(gen_addr(out, first));
      const bf16 right = s_bf16_ld_g(gen_addr(out, second));
      union { float f; unsigned int u; } a, b;
      a.f = s_f32_ld_g(gen_addr(out, full_f32));
      b.f = s_f32_ld_g(gen_addr(out, split_f32));
      const unsigned int delta = a.u >= b.u ? a.u - b.u : b.u - a.u;
      const unsigned int fraction = a.u & 65535;
      const unsigned int distance = fraction >= 32768 ? fraction - 32768 : 32768 - fraction;
      // This window is an empirical trigger, not a proof of the MME error.
      const unsigned int window = delta < 8 ? 64 : delta * 8;
      if (left == right && (a.u >> 31) == (b.u >> 31) && delta < 4096 && distance > window) {
        v_bf16_st_tnsr_partial(out, output, (bfloat128)left, 0, 0);
        continue;
      }
      float128 sum = {0.f, 0.f}, compensation = {0.f, 0.f};
      for (int offset = 0; offset < width; offset += 128) {
        const int count = s_i32_min(width - offset, 128) - 1;
        int5 x_pos = {offset, row, 0, 0, 0};
        int5 w_pos = {offset, column, 0, 0, 0};
        float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(x_pos, input, (char)count, 0));
        float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(w_pos, weight, (char)count, 0));
        float128 y = {x.v1 * w.v1 - compensation.v1, x.v2 * w.v2 - compensation.v2};
        float128 next = {sum.v1 + y.v1, sum.v2 + y.v2};
        compensation = (float128){(next.v1 - sum.v1) - y.v1, (next.v2 - sum.v2) - y.v2};
        sum = next;
      }
      float128 p = reduce_pair(sum.v1, -compensation.v1);
      float128 q = reduce_pair(sum.v2, -compensation.v2);
      float128 total = add_pair(p.v1, p.v2, q.v1, q.v2);
      float64 result = total.v1 + total.v2;
      float64 z = result - total.v1;
      float64 remainder = (total.v1 - (result - z)) + (total.v2 - z);
      uint64 bits = (uint64)result;
      uint64 up = v_u32_sel_neq_u32_b(bits & 0x80000000, 0, bits - 1, bits + 1);
      uint64 down = v_u32_sel_neq_u32_b(bits & 0x80000000, 0, bits + 1, bits - 1);
      uint64 adjustment = v_u32_sel_grt_f32_b(remainder, 0.f, up, down);
      adjustment = v_u32_sel_neq_f32_b(remainder, 0.f, adjustment, bits);
      uint64 corrected = v_u32_sel_eq_u32_b(bits & 0xffff, 0x8000, adjustment, bits);
      float128 both = {(float64)corrected, (float64)corrected};
      v_bf16_st_tnsr_partial(out, output, convert_float128_to_bfloat128(both, SW_LINEAR | SW_RHNE), 0, 0);
    }
  }
}
