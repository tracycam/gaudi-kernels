// SPDX-License-Identifier: Apache-2.0
// Decode the existing block-quantized activation into compensated BF16 terms.
void main(tensor input, tensor scales, tensor high, tensor low) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  for (int row = begin[1]; row < end[1]; ++row) {
    for (int block = begin[0]; block < end[0]; ++block) {
      int5 src = {0, row, block, 0, 0};
      int5 dst = {block * 128, row, 0, 0, 0};
      float scale = s_f32_ld_g(gen_addr(src, scales));
      minifloat256 packed = v_f8_ld_tnsr_b(src, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0);
      bfloat128 decoded = v_convert_f8_to_bf16_b(packed);
      float128 value = convert_bfloat128_to_float128(decoded, SW_LINEAR);
      value.v1 *= scale; value.v2 *= scale;
      bfloat128 first = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
      float128 widened = convert_bfloat128_to_float128(first, SW_LINEAR);
      value.v1 -= widened.v1; value.v2 -= widened.v2;
      bfloat128 second = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
      v_bf16_st_tnsr(dst, high, first);
      v_bf16_st_tnsr(dst, low, second);
    }
  }
}
