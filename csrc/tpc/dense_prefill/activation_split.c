// SPDX-License-Identifier: Apache-2.0
// Preserve the BF16 rounding residual for compensated MME multiplication.
void main(tensor input, tensor high, tensor low) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int chunk = begin[0]; chunk < end[0]; ++chunk) {
      int5 p = {chunk * 128, row, 0, 0, 0};
      float128 value;
      value.v1 = v_f32_ld_tnsr_b(p, input);
      p[0] += 64;
      value.v2 = v_f32_ld_tnsr_b(p, input);
      bfloat128 first = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
      float128 widened = convert_bfloat128_to_float128(first, SW_LINEAR);
      value.v1 -= widened.v1;
      value.v2 -= widened.v2;
      bfloat128 second = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
      p[0] -= 64;
      v_bf16_st_tnsr(p, high, first);
      v_bf16_st_tnsr(p, low, second);
    }
  }
}
