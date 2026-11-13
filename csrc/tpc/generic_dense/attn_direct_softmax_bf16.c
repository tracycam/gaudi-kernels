// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Coarsened 2-row LUT-free single-block causal softmax for 3D {keys, rows, heads}
// and 4D {keys, rows, group, kv_heads} KV buckets up to 512.
// Uses 2-row tensor-load pipelining across all key buckets {128, 256, 384, 512}
// with 2-FMA Cody-Waite expf and 3-step Newton-Raphson rcp (0 SFU LUTs).
void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, float scale, float softcap, int sliding_window) {
  const int block_rows = 16;
#define ATTN_SOFTMAX_EMIT_STATS 0
#include "attn_softmax_body.h"
#undef ATTN_SOFTMAX_EMIT_STATS
}
