// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"
#include "../moe_router/precise_ratio.h"

// Correctness-first online causal GQA over NHD slots and a request/token map.
void main(tensor query, tensor key_pool, tensor value_pool, tensor request_ids,
          tensor positions, tensor request_to_token, tensor metadata, tensor output,
          int head_dim, float scale) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, metadata));
  const int q_heads = get_dim_size(query, 0) / head_dim;
  const int kv_heads = get_dim_size(key_pool, 0) / head_dim;
  const int ratio = q_heads / kv_heads;
  const float negative_infinity = -1.0f / 0.0f;
  for (int row = begin[1]; row < end[1]; ++row) {
    if (row >= valid_rows) continue;
    int5 index = {row, 0, 0, 0, 0};
    const int request = s_i32_ld_g(gen_addr(index, request_ids));
    const int position = s_i32_ld_g(gen_addr(index, positions));
    for (int head = begin[0]; head < end[0]; ++head) {
      int5 qpos = {head * head_dim, row, 0, 0, 0};
      const bfloat128 q = v_bf16_ld_tnsr_partial_b(qpos, query, (char)(head_dim - 1), 0);
      const int kv_offset = (head / ratio) * head_dim;
      float64 maximum = negative_infinity, denominator = 0.f;
      float128 accumulator = {0.f, 0.f};
      for (int token = 0; token <= position; ++token) {
        int5 map_pos = {token, request, 0, 0, 0};
        const int slot = s_i32_ld_g(gen_addr(map_pos, request_to_token));
        int5 kv_pos = {kv_offset, slot, 0, 0, 0};
        const bfloat128 k = v_bf16_ld_tnsr_partial_b(kv_pos, key_pool, (char)(head_dim - 1), 0);
        const bfloat128 v = v_bf16_ld_tnsr_partial_b(kv_pos, value_pool, (char)(head_dim - 1), 0);
        const float128 product = v_bf16_mac_acc32_b(q, k, (float128){0.f, 0.f});
        const float64 score = v_f32_reduce_add(product.v1 + product.v2) * scale;
        const float64 next_max = v_f32_max_b(maximum, score);
        const float64 old_weight = tpc_exp_f32_safe(maximum - next_max);
        const float64 new_weight = tpc_exp_f32_safe(score - next_max);
        const float128 wide_value = v_convert_bf16_to_f32_all_b(v);
        accumulator.v1 = v_f32_mac_b(wide_value.v1, new_weight, accumulator.v1 * old_weight);
        accumulator.v2 = v_f32_mac_b(wide_value.v2, new_weight, accumulator.v2 * old_weight);
        denominator = v_f32_mac_b(denominator, old_weight, new_weight);
        maximum = next_max;
      }
      float128 result = {router_precise_ratio(accumulator.v1, denominator),
                         router_precise_ratio(accumulator.v2, denominator)};
      v_bf16_st_tnsr_partial(qpos, output, convert_float128_to_bfloat128(result, SW_RHNE),
                            (char)(head_dim - 1), 0);
    }
  }
}
