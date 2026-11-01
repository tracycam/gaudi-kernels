// SPDX-License-Identifier: Apache-2.0
// Split the scaled FP8 weight into two BF16 terms for compensated MME.
void main(tensor input, tensor scales, tensor high, tensor low) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int rows = get_dim_size(input, 1);
  for (int nb = begin[1]; nb < end[1]; ++nb) {
    for (int kb = begin[0]; kb < end[0]; ++kb) {
      int5 sp = {kb, nb, 0, 0, 0};
      float scale = s_f32_ld_g(gen_addr(sp, scales));
      #pragma loop_unroll(4)
      for (int r = 0; r < 128; ++r) {
        int row = nb * 128 + r;
        bool live = row < rows;
        int5 src = {0, row, kb, 0, 0}, dst = {kb * 128, row, 0, 0, 0};
        minifloat256 packed = v_f8_ld_tnsr_b(src, input, SW_UNPACK | SW_UNPCK_8_TO_16, 0, live);
        bfloat128 decoded = v_convert_f8_to_bf16_b(packed);
        float128 value = convert_bfloat128_to_float128(decoded, SW_LINEAR);
        value.v1 *= scale; value.v2 *= scale;
        bfloat128 first = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
        float128 widened = convert_bfloat128_to_float128(first, SW_LINEAR);
        value.v1 -= widened.v1; value.v2 -= widened.v2;
        bfloat128 second = convert_float128_to_bfloat128(value, SW_LINEAR | SW_RHNE);
        v_bf16_st_tnsr(dst, high, first, 0, live);
        v_bf16_st_tnsr(dst, low, second, 0, live);
      }
    }
  }
}
