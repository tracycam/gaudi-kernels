// SPDX-License-Identifier: Apache-2.0
// Treat int64 inputs as pairs of int32 lanes; reject values outside int32 range.
static inline int read_control(tensor value, int row) {
  int5 low_pos = {0, row, 0, 0, 0};
  const int low = s_i32_ld_g(gen_addr(low_pos, value));
  if (get_dim_size(value, 0) == 2) {
    int5 high_pos = {1, row, 0, 0, 0};
    if (s_i32_ld_g(gen_addr(high_pos, value)) != 0) return -1;
  }
  return low;
}

// Validate scheduler metadata once per batch and expand request IDs on device.
void main(tensor locations, tensor batch_requests, tensor positions, tensor mapping,
          tensor header, tensor info, tensor row_locations, tensor row_requests,
          tensor row_positions, tensor status, int slots) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0, 0, 0, 0, 0};
  const int valid_rows = s_i32_ld_g(gen_addr(zero, header));
  const int batch_count = s_i32_ld_g(gen_addr(zero, info));
  const int bucket = get_dim_size(locations, 1);
  const int requests = get_dim_size(mapping, 1);
  const int context = get_dim_size(mapping, 0);
  for (int row = begin[0]; row < end[0]; ++row) {
    int5 row_pos = {row, 0, 0, 0, 0};
    if (row >= valid_rows) {
      s_i32_st_g(gen_addr(row_pos, row_locations), 0);
      s_i32_st_g(gen_addr(row_pos, row_requests), 0);
      s_i32_st_g(gen_addr(row_pos, row_positions), 0);
      s_i32_st_g(gen_addr(row_pos, status), 0);
      continue;
    }
    int5 batch_pos = {1 + row, 0, 0, 0, 0};
    int5 expected_pos = {1 + bucket + row, 0, 0, 0, 0};
    const int batch = s_i32_ld_g(gen_addr(batch_pos, info));
    const int expected = s_i32_ld_g(gen_addr(expected_pos, info));
    int error = 0, request = 0;
    const int slot = read_control(locations, row);
    const int position = read_control(positions, row);
    if (batch < 0 || batch >= batch_count || slot < 0 || slot >= slots ||
        position < 0 || position >= context || position != expected) {
      error = 1;
    } else {
      request = read_control(batch_requests, batch);
      if (request < 0 || request >= requests) error = 2;
      else {
        int5 map_pos = {position, request, 0, 0, 0};
        if (s_i32_ld_g(gen_addr(map_pos, mapping)) != slot) error = 3;
      }
    }
    for (int other = 0; other < valid_rows && !error; ++other) {
      if (other == row) continue;
      if (read_control(locations, other) == slot) error = 4;
    }
    for (int item = 0; item < batch_count && !error; ++item) {
      int5 prefix_pos = {1 + 2 * bucket + item, 0, 0, 0, 0};
      int5 length_pos = {1 + 3 * bucket + item, 0, 0, 0, 0};
      const int other_request = read_control(batch_requests, item);
      const int prefix = s_i32_ld_g(gen_addr(prefix_pos, info));
      const int length = s_i32_ld_g(gen_addr(length_pos, info));
      if (other_request < 0 || other_request >= requests || prefix < 0 ||
          length <= 0 || prefix >= context || length > context - prefix) {
        error = 5;
        break;
      }
      for (int token = 0; token < prefix + length; ++token) {
        int5 map_pos = {token, other_request, 0, 0, 0};
        const int mapped = s_i32_ld_g(gen_addr(map_pos, mapping));
        if (mapped < 0 || mapped >= slots) { error = 6; break; }
        if (mapped == slot && (other_request != request || token != position)) { error = 7; break; }
      }
    }
    s_i32_st_g(gen_addr(row_pos, row_locations), error ? 0 : slot);
    s_i32_st_g(gen_addr(row_pos, row_requests), error ? 0 : request);
    s_i32_st_g(gen_addr(row_pos, row_positions), error ? 0 : position);
    s_i32_st_g(gen_addr(row_pos, status), error);
  }
}
