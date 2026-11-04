// SPDX-License-Identifier: Apache-2.0
// Both outputs consume the same BF16 residual boundary.
void main(tensor x, tensor residual, tensor gamma, tensor residual_out, tensor norm_out,
          float epsilon) {
  const int width = get_dim_size(x, 0);
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  bfloat128 saved[64];
  for (int row = begin[0]; row < end[0]; ++row) {
    float128 ss = {0, 0};
    for (int k = 0; k < width; k += 128) {
      int5 pos = {k, row, 0, 0, 0};
      int count = s_i32_min(width - k, 128) - 1;
      float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(pos, x, (char)count, 0));
      float128 b = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(pos, residual, (char)count, 0));
      float128 sum = {a.v1 + b.v1, a.v2 + b.v2};
      bfloat128 rounded = convert_float128_to_bfloat128(sum, SW_RHNE);
      saved[k / 128] = rounded;
      v_bf16_st_tnsr_partial(pos, residual_out, rounded, (char)count, 0);
      ss = v_bf16_mac_acc32_b(rounded, rounded, ss);
    }
    float64 sum = v_f32_reduce_add(ss.v1 + ss.v2);
    float64 inverse = v_rsqrt_f32(sum * (1.0f / (float)width) + epsilon);
    for (int k = 0; k < width; k += 128) {
      int5 pos = {k, row, 0, 0, 0}, g = {k, 0, 0, 0, 0};
      int count = s_i32_min(width - k, 128) - 1;
      float128 value = v_convert_bf16_to_f32_all_b(saved[k / 128]);
      float128 weight = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(g, gamma, (char)count, 0));
      float128 normalized = {value.v1 * inverse * weight.v1, value.v2 * inverse * weight.v2};
      v_bf16_st_tnsr_partial(pos, norm_out, convert_float128_to_bfloat128(normalized, SW_RHNE), (char)count, 0);
    }
  }
}
