// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

void main(tensor input, tensor output, int approximate) {
  const int hidden = get_dim_size(output, 0);
  const int rows = get_dim_size(output, 1);
  const int chunks = (hidden + 127) / 128;
  const int total = rows * chunks;

  const int tid = get_index_space_offset()[0];
  const int ntid = get_index_space_size()[0];
  const int per = total / ntid;
  const int rem = total % ntid;
  const int start = tid * per + s_i32_min(tid, rem);
  const int end = start + per + (tid < rem ? 1 : 0);

  if (approximate != 0) {
    for (int idx = start; idx < end; ++idx) {
      const int row = idx / chunks;
      const int col = (idx - row * chunks) * 128;
      const char cnt = (char)(s_i32_min(128, hidden - col) - 1);
      int5 gpos = {col, row, 0, 0, 0};
      int5 upos = {hidden + col, row, 0, 0, 0};
      float128 gate = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(gpos, input, cnt, 0), SW_LINEAR);
      float128 up = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(upos, input, cnt, 0), SW_LINEAR);
      float128 out = {
          tpc_gelu_tanh_f32_safe(gate.v1) * up.v1,
          tpc_gelu_tanh_f32_safe(gate.v2) * up.v2,
      };
      v_bf16_st_tnsr_partial(gpos, output, convert_float128_to_bfloat128(out, SW_LINEAR | SW_RHNE), cnt, 0);
    }
  } else {
    for (int idx = start; idx < end; ++idx) {
      const int row = idx / chunks;
      const int col = (idx - row * chunks) * 128;
      const char cnt = (char)(s_i32_min(128, hidden - col) - 1);
      int5 gpos = {col, row, 0, 0, 0};
      int5 upos = {hidden + col, row, 0, 0, 0};
      float128 gate = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(gpos, input, cnt, 0), SW_LINEAR);
      float128 up = convert_bfloat128_to_float128(v_bf16_ld_tnsr_partial_b(upos, input, cnt, 0), SW_LINEAR);
      float128 out = {
          tpc_gelu_erf_f32_safe(gate.v1) * up.v1,
          tpc_gelu_erf_f32_safe(gate.v2) * up.v2,
      };
      v_bf16_st_tnsr_partial(gpos, output, convert_float128_to_bfloat128(out, SW_LINEAR | SW_RHNE), cnt, 0);
    }
  }
}
