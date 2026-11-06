// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"
#include "../moe_router/precise_ratio.h"

// Online FP32 causal softmax with one BF16 probability output.
void main(tensor scores, tensor positions, tensor metadata, tensor probabilities,
          int kv_heads, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, metadata));
  const int context = get_dim_size(scores, 0);
  const float negative_infinity = -1.0f / 0.0f;
  const uint64 lane = read_lane_id_4b_b();
  for (int group = begin[1]; group < end[1]; ++group) {
    const int row = group / kv_heads;
    if (row >= valid_rows) continue;
    int5 row_pos = {row, 0, 0, 0, 0};
    const int position = s_i32_ld_g(gen_addr(row_pos, positions));
    for (int head = begin[0]; head < end[0]; ++head) {
      float64 maximum = negative_infinity, sum = 0.f;
      for (int start = 0; start < context && start <= position; start += 128) {
        const int available0 = s_i32_min(64, context - start);
        const int available1 = s_i32_min(64, s_i32_max(0, context - start - 64));
        const unsigned valid0 = s_i32_min(available0, position + 1 - start);
        const unsigned valid1 = s_i32_min(available1, s_i32_max(0, position + 1 - start - 64));
        int5 pos = {start, head, group, 0, 0};
        float64 a = v_f32_ld_tnsr_partial_b(pos, scores, (char)(available0 - 1), 0) * scale;
        float64 b = negative_infinity;
        if (available1) {
          pos[0] += 64;
          b = v_f32_ld_tnsr_partial_b(pos, scores, (char)(available1 - 1), 0) * scale;
        }
        a = v_f32_sel_less_u32_b(lane, valid0, a, negative_infinity);
        b = v_f32_sel_less_u32_b(lane, valid1, b, negative_infinity);
        const float64 next_max = v_f32_max_b(maximum, v_f32_reduce_max(v_f32_max_b(a, b)));
        const float64 old_weight = tpc_exp_f32_safe(maximum - next_max);
        a = v_f32_sel_less_u32_b(lane, valid0, tpc_exp_f32_safe(a - next_max), 0.f);
        b = v_f32_sel_less_u32_b(lane, valid1, tpc_exp_f32_safe(b - next_max), 0.f);
        sum = v_f32_mac_b(sum, old_weight, v_f32_reduce_add(a + b));
        maximum = next_max;
      }
      for (int start = 0; start < context; start += 128) {
        const int available0 = s_i32_min(64, context - start);
        const int available1 = s_i32_min(64, s_i32_max(0, context - start - 64));
        const unsigned valid0 = s_i32_min(available0, s_i32_max(0, position + 1 - start));
        const unsigned valid1 = s_i32_min(available1, s_i32_max(0, position + 1 - start - 64));
        int5 pos = {start, head, group, 0, 0};
        float64 a = v_f32_ld_tnsr_partial_b(pos, scores, (char)(available0 - 1), 0) * scale;
        float64 b = negative_infinity;
        if (available1) {
          pos[0] += 64;
          b = v_f32_ld_tnsr_partial_b(pos, scores, (char)(available1 - 1), 0) * scale;
        }
        a = v_f32_sel_less_u32_b(lane, valid0, tpc_exp_f32_safe(a - maximum), 0.f);
        b = v_f32_sel_less_u32_b(lane, valid1, tpc_exp_f32_safe(b - maximum), 0.f);
        float128 pair = {router_precise_ratio(a, sum), router_precise_ratio(b, sum)};
        bfloat128 high = convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE);
        pos[0] = start;
        const char count = (char)(s_i32_min(128, context - start) - 1);
        v_bf16_st_tnsr_partial(pos, probabilities, high, count, 0);
      }
    }
  }
}
