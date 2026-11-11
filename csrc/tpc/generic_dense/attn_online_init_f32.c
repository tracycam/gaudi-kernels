// SPDX-License-Identifier: Apache-2.0

// Initialize online softmax running state directly from tile 0 without
// materializing zero constant_f32 tensors (supports 3D and 4D MME layouts).
void main(tensor tile_max, tensor tile_sum, tensor tile_output,
          tensor new_max, tensor new_sum, tensor new_output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  const int width = get_dim_size(tile_output, 0), rows = get_dim_size(tile_output, 1);
  const float ni = -1.0f / 0.0f;
  for (int kv_h = begin[3]; kv_h < end[3]; ++kv_h) {
    for (int head = begin[0]; head < end[0]; ++head) {
      for (int block = begin[1]; block < end[1]; ++block) {
        const int first = block * 64;
        int5 stat = {first, head, kv_h, 0, 0};
        float64 m1 = v_f32_ld_tnsr_b(stat, tile_max), l1 = v_f32_ld_tnsr_b(stat, tile_sum);
        float64 m = v_f32_sel_grt_f32_b(l1, 0.f, m1, (float64)ni);
        float64 bl1 = 0.f + v_f32_sel_grt_f32_b(l1, 0.f, l1, 0.f);
        v_f32_st_tnsr(stat, new_max, m);
        v_f32_st_tnsr(stat, new_sum, bl1);
        const int last = s_i32_min(rows - first, 64);
        int item = 0;
        if (width == 128) {
          for (; item + 1 < last; item += 2) {
            float64 bb0 = v_broadcast_element_f32(bl1, item);
            float64 bb1 = v_broadcast_element_f32(bl1, item + 1);
            int5 p0 = {0, first + item, head, kv_h, 0};
            int5 p1 = {0, first + item + 1, head, kv_h, 0};
            float64 y0_lo = v_f32_ld_tnsr_b(p0, tile_output); p0[0] = 64;
            float64 y0_hi = v_f32_ld_tnsr_b(p0, tile_output);
            float64 y1_lo = v_f32_ld_tnsr_b(p1, tile_output); p1[0] = 64;
            float64 y1_hi = v_f32_ld_tnsr_b(p1, tile_output);
            p0[0] = 0;  v_f32_st_tnsr(p0, new_output, 0.f + bb0 * y0_lo);
            p0[0] = 64; v_f32_st_tnsr(p0, new_output, 0.f + bb0 * y0_hi);
            p1[0] = 0;  v_f32_st_tnsr(p1, new_output, 0.f + bb1 * y1_lo);
            p1[0] = 64; v_f32_st_tnsr(p1, new_output, 0.f + bb1 * y1_hi);
          }
          for (; item < last; ++item) {
            float64 bb = v_broadcast_element_f32(bl1, item);
            int5 p = {0, first + item, head, kv_h, 0};
            float64 y_lo = v_f32_ld_tnsr_b(p, tile_output); p[0] = 64;
            float64 y_hi = v_f32_ld_tnsr_b(p, tile_output);
            p[0] = 0;  v_f32_st_tnsr(p, new_output, 0.f + bb * y_lo);
            p[0] = 64; v_f32_st_tnsr(p, new_output, 0.f + bb * y_hi);
          }
        } else {
          for (; item < last; ++item) {
            float64 bb = v_broadcast_element_f32(bl1, item);
            int5 p = {0, first + item, head, kv_h, 0};
            float64 y = v_f32_ld_tnsr_b(p, tile_output);
            v_f32_st_tnsr(p, new_output, 0.f + bb * y);
          }
        }
      }
    }
  }
}
