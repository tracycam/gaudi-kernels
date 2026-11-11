// SPDX-License-Identifier: Apache-2.0

// Finalize online softmax running state into BF16 (supports 3D and 4D layouts).
void main(tensor sums, tensor accumulator, tensor result) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(accumulator, 0), rows = get_dim_size(accumulator, 1);
  for (int kv_h = begin[3]; kv_h < end[3]; ++kv_h) {
    for (int head = begin[0]; head < end[0]; ++head) {
      for (int block = begin[1]; block < end[1]; ++block) {
        const int first = block * 64;
        int5 stat = {first, head, kv_h, 0, 0};
        float64 denominator = v_f32_ld_tnsr_b(stat, sums);
        float64 safe = v_f32_sel_grt_f32_b(denominator, 0.f, denominator, 1.f);
        float64 inverse = v_f32_sel_grt_f32_b(denominator, 0.f, v_reciprocal_f32(safe), 0.f);
        const int last = s_i32_min(rows - first, 64);
        int item = 0;
        if (width == 128) {
          for (; item + 1 < last; item += 2) {
            float64 r0 = v_broadcast_element_f32(inverse, item);
            float64 r1 = v_broadcast_element_f32(inverse, item + 1);
            int5 p0 = {0, first + item, head, kv_h, 0};
            int5 p1 = {0, first + item + 1, head, kv_h, 0};
            float64 a0 = v_f32_ld_tnsr_b(p0, accumulator); p0[0] = 64;
            float64 b0 = v_f32_ld_tnsr_b(p0, accumulator); p0[0] = 0;
            float64 a1 = v_f32_ld_tnsr_b(p1, accumulator); p1[0] = 64;
            float64 b1 = v_f32_ld_tnsr_b(p1, accumulator); p1[0] = 0;
            float128 pair0 = {a0 * r0, b0 * r0};
            float128 pair1 = {a1 * r1, b1 * r1};
            v_bf16_st_tnsr(p0, result, convert_float128_to_bfloat128(pair0, SW_LINEAR | SW_RHNE));
            v_bf16_st_tnsr(p1, result, convert_float128_to_bfloat128(pair1, SW_LINEAR | SW_RHNE));
          }
          for (; item < last; ++item) {
            float64 r = v_broadcast_element_f32(inverse, item);
            int5 p = {0, first + item, head, kv_h, 0};
            float64 a = v_f32_ld_tnsr_b(p, accumulator); p[0] = 64;
            float64 b = v_f32_ld_tnsr_b(p, accumulator); p[0] = 0;
            float128 pair = {a * r, b * r};
            v_bf16_st_tnsr(p, result, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
          }
        } else {
          for (; item < last; ++item) {
            float64 r = v_broadcast_element_f32(inverse, item);
            int5 p = {0, first + item, head, kv_h, 0};
            float128 pair = {v_f32_ld_tnsr_b(p, accumulator) * r, 0.f};
            bfloat128 output = convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE);
            v_bf16_st_tnsr_partial(p, result, output, 63, 0);
          }
        }
      }
    }
  }
}
