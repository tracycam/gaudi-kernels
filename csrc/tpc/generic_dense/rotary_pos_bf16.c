// SPDX-License-Identifier: Apache-2.0
#define ROTARY_APPLY_PAIR_AND_STORE(IN_TNSR, OUT_TNSR, COL_BASE, ROW, POS, HEAD_DIM, ROTARY_DIM, IS_NEOX) \
  do { \
    const int _half = (ROTARY_DIM) >> 1; \
    if ((IS_NEOX) != 0) { \
      for (int _d = 0; _d < _half; _d += 64) { \
        const int _cnt = s_i32_min(64, _half - _d); \
        const char _ccnt = (char)(_cnt - 1); \
        int5 _p0 = {(COL_BASE) + _d, (ROW), 0, 0, 0}; \
        int5 _p1 = {(COL_BASE) + _half + _d, (ROW), 0, 0, 0}; \
        int5 _pc = {_d, (POS), 0, 0, 0}; \
        int5 _ps = {_half + _d, (POS), 0, 0, 0}; \
        float64 _x0 = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_p0, (IN_TNSR), _ccnt, 0), SW_LINEAR).v1; \
        float64 _x1 = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_p1, (IN_TNSR), _ccnt, 0), SW_LINEAR).v1; \
        float64 _c  = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_pc, cache, _ccnt, 0), SW_LINEAR).v1; \
        float64 _s  = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_ps, cache, _ccnt, 0), SW_LINEAR).v1; \
        float128 _y0 = {_x0 * _c - _x1 * _s, 0.0f}; \
        float128 _y1 = {_x1 * _c + _x0 * _s, 0.0f}; \
        v_bf16_st_tnsr_partial(_p0, (OUT_TNSR), convert_float128_to_bfloat128(_y0, SW_LINEAR | SW_RHNE), _ccnt, 0); \
        v_bf16_st_tnsr_partial(_p1, (OUT_TNSR), convert_float128_to_bfloat128(_y1, SW_LINEAR | SW_RHNE), _ccnt, 0); \
      } \
    } else { \
      for (int _d = 0; _d < _half; _d += 64) { \
        const int _cnt = s_i32_min(64, _half - _d); \
        const char _ccnt = (char)(_cnt - 1); \
        const char _ccnt2 = (char)((_cnt << 1) - 1); \
        int5 _px = {(COL_BASE) + (_d << 1), (ROW), 0, 0, 0}; \
        int5 _pc = {_d, (POS), 0, 0, 0}; \
        int5 _ps = {_half + _d, (POS), 0, 0, 0}; \
        float128 _x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(_px, (IN_TNSR), _ccnt2, 0)); \
        float64 _c = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_pc, cache, _ccnt, 0), SW_LINEAR).v1; \
        float64 _s = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(_ps, cache, _ccnt, 0), SW_LINEAR).v1; \
        float128 _y = {_x.v1 * _c - _x.v2 * _s, _x.v2 * _c + _x.v1 * _s}; \
        v_bf16_st_tnsr_partial(_px, (OUT_TNSR), convert_float128_to_bfloat128(_y, SW_RHNE), _ccnt2, 0); \
      } \
    } \
    for (int _d = (ROTARY_DIM); _d < (HEAD_DIM); _d += 128) { \
      const char _ccnt = (char)(s_i32_min(128, (HEAD_DIM) - _d) - 1); \
      int5 _pt = {(COL_BASE) + _d, (ROW), 0, 0, 0}; \
      v_bf16_st_tnsr_partial(_pt, (OUT_TNSR), v_bf16_ld_tnsr_partial_b(_pt, (IN_TNSR), _ccnt, 0), _ccnt, 0); \
    } \
  } while (0)

void main(tensor query, tensor key, tensor positions, tensor cache,
          tensor query_out, tensor key_out,
          int head_dim, int rotary_dim, int is_neox) {
  const int q_width = get_dim_size(query, 0);
  const int rows = get_dim_size(query, 1);
  const int k_width = get_dim_size(key, 0);
  const int cache_rows = get_dim_size(cache, 1);
  const int q_heads = q_width / head_dim;
  const int k_heads = k_width / head_dim;
  const int total_heads = q_heads + k_heads;
  const int total_work = rows * total_heads;

  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const int per = total_work / ntid;
  const int rem = total_work % ntid;
  const int start = tid * per + s_i32_min(tid, rem);
  const int end = start + per + (tid < rem ? 1 : 0);

  int prev_row = -1;
  int pos = 0;
  for (int idx = start; idx < end; ++idx) {
    const int row = idx / total_heads;
    const int head = idx - row * total_heads;
    if (row != prev_row) {
      prev_row = row;
      int5 pos_coord = {row, 0, 0, 0, 0};
      const int raw_pos = s_i32_ld_g(gen_addr(pos_coord, positions));
      pos = s_i32_max(0, s_i32_min(raw_pos, cache_rows - 1));
    }
    if (head < q_heads) {
      ROTARY_APPLY_PAIR_AND_STORE(query, query_out, head * head_dim, row, pos, head_dim, rotary_dim, is_neox);
    } else {
      ROTARY_APPLY_PAIR_AND_STORE(key, key_out, (head - q_heads) * head_dim, row, pos, head_dim, rotary_dim, is_neox);
    }
  }
}
