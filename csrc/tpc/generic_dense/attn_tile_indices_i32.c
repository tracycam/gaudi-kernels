// SPDX-License-Identifier: Apache-2.0
// Each task expands 64 consecutive request-map slots for one KV head/kind.
// Task fields: query row, first key position, valid lanes, head/kind offset.
void main(tensor mapping, tensor requests, tensor tasks, tensor header, tensor indices) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int5 zero = {0,0,0,0,0};
  const int context = s_i32_ld_g(gen_addr(zero, header));
  const uint64 lanes = read_lane_id_4b_b();
  for (int task = begin[0]; task < end[0]; ++task) {
    int5 p = {0,task,0,0,0};
    const int row = s_i32_ld_g(gen_addr(p, tasks)); p[0] = 1;
    const int first = s_i32_ld_g(gen_addr(p, tasks)); p[0] = 2;
    const int valid = s_i32_ld_g(gen_addr(p, tasks)); p[0] = 3;
    const int offset = s_i32_ld_g(gen_addr(p, tasks));
    int64 result = 0;
    if (valid > 0) {
      int5 request_pos = {row,0,0,0,0};
      const int request = s_i32_ld_g(gen_addr(request_pos, requests));
      int5 map_pos = {request * context + first,0,0,0,0};
      int64 slots = v_i32_ld_tnsr_b(map_pos, mapping);
      result = v_i32_sel_less_u32_b(lanes, valid, slots, 0) + offset;
    }
    int5 output_pos = {0,task,0,0,0};
    v_i32_st_tnsr(output_pos, indices, result);
  }
}
