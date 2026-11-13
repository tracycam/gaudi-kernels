// SPDX-License-Identifier: Apache-2.0
#include "../common/tpc_math_safe.h"

// Coarsened 2-row LUT-free causal tile softmax for 3D {keys, rows, heads} and
// 4D {keys, rows, group, kv_heads} KV buckets up to 512.
// Uses 2-FMA Cody-Waite degree-6 Estrin expf and 3-step Newton-Raphson rcp (0
// SFU LUTs, full 64 KiB VLM unlocked) with 2-row tensor-load pipelining.
void main(tensor scores, tensor positions, tensor header,
          tensor probabilities, tensor tile_max, tensor tile_sum,
          float scale, int block_rows, float softcap, int sliding_window) {
#define ATTN_SOFTMAX_EMIT_STATS 1
#include "attn_softmax_body.h"
#undef ATTN_SOFTMAX_EMIT_STATS
}
