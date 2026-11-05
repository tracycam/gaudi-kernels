// SPDX-License-Identifier: Apache-2.0
// Refine only outputs whose two FP32 MME reduction trees round differently.
void main(tensor input, tensor weight, tensor first, tensor second, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(input, 0);
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int column = begin[0]; column < end[0]; ++column) {
      int5 out = {column, row, 0, 0, 0};
      const bf16 left = s_bf16_ld_g(gen_addr(out, first));
      const bf16 right = s_bf16_ld_g(gen_addr(out, second));
      if (left == right) {
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
      float64 total = v_f32_reduce_add(sum.v1) + v_f32_reduce_add(sum.v2);
      float64 center = total * 0.0078125f;
      float64 residual = v_f32_reduce_add(sum.v1 - center) + v_f32_reduce_add(sum.v2 - center);
      float64 result = total + residual;
      float128 both = {result, result};
      v_bf16_st_tnsr_partial(out, output, convert_float128_to_bfloat128(both, SW_LINEAR | SW_RHNE), 0, 0);
    }
  }
}
