// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

extern "C" unsigned char _binary_attn_tile_softmax_bf16_o_start, _binary_attn_tile_softmax_bf16_o_end;
extern "C" unsigned char _binary_attn_online_merge_f32_o_start, _binary_attn_online_merge_f32_o_end;
extern "C" unsigned char _binary_attn_online_finalize_bf16_o_start, _binary_attn_online_finalize_bf16_o_end;
extern "C" unsigned char _binary_kv_store_head_major_bf16_o_start, _binary_kv_store_head_major_bf16_o_end;
extern "C" unsigned char _binary_kv_store_head_major64_bf16_o_start, _binary_kv_store_head_major64_bf16_o_end;
extern "C" unsigned char _binary_attn_decode_exp_f32_o_start, _binary_attn_decode_exp_f32_o_end;
extern "C" unsigned char _binary_attn_decode_normalize_bf16_o_start, _binary_attn_decode_normalize_bf16_o_end;
extern "C" unsigned char _binary_attn_direct_softmax_bf16_o_start, _binary_attn_direct_softmax_bf16_o_end;
extern "C" unsigned char _binary_attn_tile_indices_i32_o_start, _binary_attn_tile_indices_i32_o_end;

namespace generic_attention {
using namespace tpc_lib_api;
constexpr const char* kSoftmax = "gk_attn_tile_softmax_bf16_v1";
constexpr const char* kMerge = "gk_attn_online_merge_f32_v1";
constexpr const char* kFinalize = "gk_attn_online_finalize_bf16_v1";
constexpr const char* kHeadStore = "gk_kv_store_head_major_bf16_v1";
constexpr const char* kDecodeExp = "gk_attn_decode_exp_f32_v1";
constexpr const char* kDecodeNormalize = "gk_attn_decode_normalize_bf16_v1";
constexpr const char* kDirectSoftmax = "gk_attn_direct_softmax_bf16_v1";
constexpr const char* kTileIndices = "gk_attn_tile_indices_i32_v1";
constexpr unsigned long kMaxScoreElements3d = 8UL * 1024 * 1024;
constexpr unsigned long kMaxScoreElements4d = 16UL * 1024 * 1024;

inline bool shape(const Tensor& tensor, TensorDataType type, std::initializer_list<unsigned long> sizes) {
  const auto& g = tensor.geometry;
  if (g.dataType != type || g.dims != sizes.size()) return false;
  unsigned dim = 0;
  for (auto size : sizes) {
    if (g.maxSizes[dim] != size || g.minSizes[dim] != size) return false;
    ++dim;
  }
  return true;
}
inline void mapped(TensorAccessPattern& access, unsigned dim, unsigned space, unsigned end, unsigned stride) {
  access.allRequired = 0;
  access.mapping[dim].indexSpaceDim = space;
  access.mapping[dim].a = stride;
  access.mapping[dim].start_b = 0;
  access.mapping[dim].end_b = end;
}
inline GlueCodeReturn binary(HabanaKernelInstantiation* output, unsigned char* start, unsigned char* end) {
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = static_cast<unsigned>(end - start);
  if (capacity < output->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, start, output->kernel.elfSize);
  return GLUE_SUCCESS;
}

inline GlueCodeReturn instantiate_tile_indices(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& mapping = input->inputTensors[0].geometry;
  const auto& requests = input->inputTensors[1].geometry;
  const auto& tasks = input->inputTensors[2].geometry;
  const auto count = tasks.maxSizes[1];
  if (mapping.dataType != DATA_I32 || mapping.dims != 1 || !mapping.maxSizes[0] ||
      mapping.minSizes[0] != 1 || requests.dataType != DATA_I32 || requests.dims != 1 ||
      !requests.maxSizes[0] || requests.minSizes[0] != requests.maxSizes[0] ||
      !count || !shape(input->inputTensors[2], DATA_I32, {4, count}) ||
      !shape(input->inputTensors[3], DATA_I32, {1}) ||
      !shape(input->outputTensors[0], DATA_I32, {64, count}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = count;
  output->indexSpaceGeometry[1] = 1;
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->inputTensorAccessPattern[1].allRequired = 1;
  mapped(output->inputTensorAccessPattern[2], 0, 1, 3, 0);
  mapped(output->inputTensorAccessPattern[2], 1, 0, 0, 1);
  output->inputTensorAccessPattern[3].allRequired = 1;
  mapped(output->outputTensorAccessPattern[0], 0, 1, 63, 0);
  mapped(output->outputTensorAccessPattern[0], 1, 0, 0, 1);
  output->kernel.paramsNr = 0;
  return binary(output, &_binary_attn_tile_indices_i32_o_start, &_binary_attn_tile_indices_i32_o_end);
}

inline GlueCodeReturn instantiate_head_store(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& source = input->inputTensors[0].geometry;
  const auto& pool = input->outputTensors[0].geometry;
  const unsigned long width = source.maxSizes[0], rows = source.maxSizes[1];
  const unsigned long d = pool.maxSizes[0], capacity = pool.maxSizes[1];
  if ((d != 64 && d != 128) || width % d) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const unsigned long heads = width / d;
  if (!heads || width % 128 || !rows || rows > 4096 || !capacity || capacity > INT_MAX ||
      !shape(input->inputTensors[0], DATA_BF16, {width, rows}) ||
      !shape(input->inputTensors[1], DATA_BF16, {width, rows}) ||
      !shape(input->inputTensors[2], DATA_I32, {rows}) ||
      !shape(input->inputTensors[3], DATA_I32, {3}) ||
      !shape(input->inputTensors[4], DATA_BF16, {d, capacity}) ||
      !shape(input->outputTensors[0], DATA_BF16, {d, capacity}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 4;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = rows;
  output->indexSpaceGeometry[2] = 1;
  output->indexSpaceGeometry[3] = 1;
  for (unsigned i : {0u, 1u}) {
    auto& access = output->inputTensorAccessPattern[i];
    mapped(access, 0, 0, d - 1, d);
    mapped(access, 1, 1, 0, 1);
  }
  mapped(output->inputTensorAccessPattern[2], 0, 1, 0, 1);
  mapped(output->inputTensorAccessPattern[3], 0, 2, 2, 0);
  for (auto* access : {&output->inputTensorAccessPattern[4], &output->outputTensorAccessPattern[0]}) {
    mapped(*access, 0, 2, d - 1, 0);
    mapped(*access, 1, 3, capacity - 1, 0);
  }
  // Only selected slots are written. Mandatory reuse prevents an SRAM slice
  // from being copied back over the rest of the persistent pool.
  auto& out = output->outputTensorAccessPattern[0];
  out.inputsReusability = 1u << 4;
  out.inputReusabilityBinding = 1;
  out.sparseAccess = 1;
  output->kernel.paramsNr = 0;
  return d == 128 ? binary(output, &_binary_kv_store_head_major_bf16_o_start,
                           &_binary_kv_store_head_major_bf16_o_end)
                  : binary(output, &_binary_kv_store_head_major64_bf16_o_start,
                           &_binary_kv_store_head_major64_bf16_o_end);
}

inline GlueCodeReturn instantiate_direct_softmax(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != sizeof(float) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float scale;
  std::memcpy(&scale, input->nodeParams.nodeParams, sizeof(scale));
  if (!std::isfinite(scale) || scale <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& data = input->inputTensors[0].geometry;
  if (data.dims == 4) {
    const unsigned long keys = data.maxSizes[0], rows = data.maxSizes[1];
    const unsigned long group = data.maxSizes[2], kv_heads = data.maxSizes[3];
    if (!keys || keys > 512 || keys % 128 || !rows || rows > 512 || !group || group > 256 ||
        !kv_heads || kv_heads > 256 || keys * rows * group * kv_heads > kMaxScoreElements4d ||
        !shape(input->inputTensors[0], DATA_F32, {keys, rows, group, kv_heads}) ||
        !shape(input->inputTensors[1], DATA_I32, {rows}) ||
        !shape(input->inputTensors[2], DATA_I32, {3}) ||
        !shape(input->outputTensors[0], DATA_BF16, {keys, rows, group, kv_heads}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned row_end = static_cast<unsigned>(std::min<unsigned long>(rows, 16) - 1);
    output->indexSpaceRank = 4;
    output->indexSpaceGeometry[0] = group;
    output->indexSpaceGeometry[1] = (rows + 15) / 16;
    output->indexSpaceGeometry[2] = 1;
    output->indexSpaceGeometry[3] = kv_heads;
    for (auto* access : {&output->inputTensorAccessPattern[0], &output->outputTensorAccessPattern[0]}) {
      mapped(*access, 0, 2, keys - 1, 0);
      mapped(*access, 1, 1, row_end, 16);
      mapped(*access, 2, 0, 0, 1);
      mapped(*access, 3, 3, 0, 1);
    }
    mapped(output->inputTensorAccessPattern[1], 0, 1, row_end, 16);
    mapped(output->inputTensorAccessPattern[2], 0, 2, 2, 0);
    output->kernel.paramsNr = 1;
    std::memcpy(output->kernel.scalarParams, &scale, sizeof(scale));
    return binary(output, &_binary_attn_direct_softmax_bf16_o_start,
                  &_binary_attn_direct_softmax_bf16_o_end);
  }
  const unsigned long keys = data.maxSizes[0], rows = data.maxSizes[1], heads = data.maxSizes[2];
  if (!keys || keys > 512 || keys % 128 || !rows || rows > 256 || !heads || heads > 256 ||
      keys * rows * heads > kMaxScoreElements3d ||
      !shape(input->inputTensors[0], DATA_F32, {keys, rows, heads}) ||
      !shape(input->inputTensors[1], DATA_I32, {rows}) ||
      !shape(input->inputTensors[2], DATA_I32, {3}) ||
      !shape(input->outputTensors[0], DATA_BF16, {keys, rows, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = (rows + 15) / 16;
  output->indexSpaceGeometry[2] = 1;
  for (auto* access : {&output->inputTensorAccessPattern[0], &output->outputTensorAccessPattern[0]}) {
    mapped(*access, 0, 2, keys - 1, 0);
    mapped(*access, 1, 1, 15, 16);
    mapped(*access, 2, 0, 0, 1);
  }
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->inputTensorAccessPattern[2].allRequired = 1;
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &scale, sizeof(scale));
  return binary(output, &_binary_attn_direct_softmax_bf16_o_start,
                &_binary_attn_direct_softmax_bf16_o_end);
}

// Long single-row softmax uses an internal FP32 exponential tensor and a
// vector of repeated sums; the graph normalizes and casts before the PV MME.
inline GlueCodeReturn instantiate_decode(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                         bool exponent) {
  if (input->inputTensorNr != (exponent ? 3u : 2u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != (exponent ? 2u : 1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if ((exponent && (input->nodeParams.nodeParamsSize != sizeof(float) || !input->nodeParams.nodeParams)) ||
      (!exponent && input->nodeParams.nodeParamsSize))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float scale = 0;
  if (exponent) {
    std::memcpy(&scale, input->nodeParams.nodeParams, sizeof(scale));
    if (!std::isfinite(scale) || scale <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  const auto& data = input->inputTensors[0].geometry;
  const unsigned long keys = data.maxSizes[0], heads = data.maxSizes[2];
  if (keys <= 512 || keys > 8192 || keys % 128 || !heads || heads > 256 ||
      !shape(input->inputTensors[0], DATA_F32, {keys, 1, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (exponent) {
    if (!shape(input->inputTensors[1], DATA_I32, {1}) ||
        !shape(input->inputTensors[2], DATA_I32, {3}) ||
        !shape(input->outputTensors[0], DATA_F32, {keys, 1, heads}) ||
        !shape(input->outputTensors[1], DATA_F32, {64, heads}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  } else if (!shape(input->inputTensors[1], DATA_F32, {64, heads}) ||
             !shape(input->outputTensors[0], DATA_BF16, {keys, 1, heads})) {
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = 1;
  output->indexSpaceGeometry[2] = 1;
  for (unsigned i = 0; i < input->inputTensorNr; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
    if (i == 0) {
      mapped(access, 0, 2, keys - 1, 0);
      mapped(access, 1, 1, 0, 0);
      mapped(access, 2, 0, 0, 1);
    } else if (!exponent && i == 1) {
      mapped(access, 0, 2, 63, 0);
      mapped(access, 1, 0, 0, 1);
    } else {
      access.allRequired = 1;
    }
  }
  for (unsigned i = 0; i < input->outputTensorNr; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    if (exponent && i == 1) {
      mapped(access, 0, 2, 63, 0);
      mapped(access, 1, 0, 0, 1);
    } else {
      mapped(access, 0, 2, keys - 1, 0);
      mapped(access, 1, 1, 0, 0);
      mapped(access, 2, 0, 0, 1);
    }
  }
  output->kernel.paramsNr = exponent ? 1 : 0;
  if (exponent) std::memcpy(output->kernel.scalarParams, &scale, sizeof(scale));
  if (exponent)
    return binary(output, &_binary_attn_decode_exp_f32_o_start, &_binary_attn_decode_exp_f32_o_end);
  return binary(output, &_binary_attn_decode_normalize_bf16_o_start,
                &_binary_attn_decode_normalize_bf16_o_end);
}

inline GlueCodeReturn instantiate_softmax(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != sizeof(float) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float scale;
  std::memcpy(&scale, input->nodeParams.nodeParams, sizeof(scale));
  if (!std::isfinite(scale) || scale <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& score = input->inputTensors[0].geometry;
  if (score.dims == 4) {
    const unsigned long keys = score.maxSizes[0], rows = score.maxSizes[1];
    const unsigned long group = score.maxSizes[2], kv_heads = score.maxSizes[3];
    const unsigned long stat_rows = (rows + 63) / 64 * 64;
    if (!keys || keys > 512 || keys % 128 || !rows || rows > 512 || !group || group > 256 ||
        !kv_heads || kv_heads > 256 || keys * rows * group * kv_heads > kMaxScoreElements4d ||
        !shape(input->inputTensors[0], DATA_F32, {keys, rows, group, kv_heads}) ||
        !shape(input->inputTensors[1], DATA_I32, {rows}) ||
        !shape(input->inputTensors[2], DATA_I32, {3}) ||
        !shape(input->outputTensors[0], DATA_BF16, {keys, rows, group, kv_heads}) ||
        !shape(input->outputTensors[1], DATA_F32, {stat_rows, group, kv_heads}) ||
        !shape(input->outputTensors[2], DATA_F32, {stat_rows, group, kv_heads}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned long block_rows = (rows % 64 != 0) ? 64UL : 16UL;
    const unsigned long row_blocks = stat_rows / block_rows;
    const unsigned row_end = static_cast<unsigned>(std::min(rows, block_rows) - 1);
    output->indexSpaceRank = 4;
    output->indexSpaceGeometry[0] = group;
    output->indexSpaceGeometry[1] = row_blocks;
    output->indexSpaceGeometry[2] = 1;
    output->indexSpaceGeometry[3] = kv_heads;
    for (auto* access : {&output->inputTensorAccessPattern[0], &output->outputTensorAccessPattern[0]}) {
      mapped(*access, 0, 2, keys - 1, 0);
      mapped(*access, 1, 1, row_end, block_rows);
      mapped(*access, 2, 0, 0, 1);
      mapped(*access, 3, 3, 0, 1);
    }
    mapped(output->inputTensorAccessPattern[1], 0, 1, row_end, block_rows);
    mapped(output->inputTensorAccessPattern[2], 0, 2, 2, 0);
    for (unsigned i : {1u, 2u}) {
      mapped(output->outputTensorAccessPattern[i], 0, 1, block_rows - 1, block_rows);
      mapped(output->outputTensorAccessPattern[i], 1, 0, 0, 1);
      mapped(output->outputTensorAccessPattern[i], 2, 3, 0, 1);
    }
    struct SoftmaxScalarParams {
      float scale;
      int32_t block_rows;
    } params{scale, static_cast<int32_t>(block_rows)};
    output->kernel.paramsNr = 2;
    std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
    return binary(output, &_binary_attn_tile_softmax_bf16_o_start,
                  &_binary_attn_tile_softmax_bf16_o_end);
  }
  const unsigned long keys = score.maxSizes[0], rows = score.maxSizes[1], heads = score.maxSizes[2];
  const unsigned long stat_rows = (rows + 63) / 64 * 64;
  // Keep each score tile bounded. Full prompts are dispatched as query tiles.
  if (!keys || keys > 512 || keys % 128 || !rows || rows > 256 || !heads || heads > 256 ||
      keys * rows * heads > kMaxScoreElements3d ||
      !shape(input->inputTensors[0], DATA_F32, {keys, rows, heads}) ||
      !shape(input->inputTensors[1], DATA_I32, {rows}) || !shape(input->inputTensors[2], DATA_I32, {3}) ||
      !shape(input->outputTensors[0], DATA_BF16, {keys, rows, heads}) ||
      !shape(input->outputTensors[1], DATA_F32, {stat_rows, heads}) ||
      !shape(input->outputTensors[2], DATA_F32, {stat_rows, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const unsigned long block_rows = (rows % 64 != 0) ? 64UL : 16UL;
  const unsigned long row_blocks = stat_rows / block_rows;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = row_blocks;
  output->indexSpaceGeometry[2] = 1;
  for (unsigned i : {0u, 1u}) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    mapped(access, 0, 2, keys - 1, 0);
    mapped(access, 1, 1, block_rows - 1, block_rows);
    mapped(access, 2, 0, 0, 1);
  }
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->inputTensorAccessPattern[2].allRequired = 1;
  for (unsigned i : {1u, 2u}) {
    auto& access = output->outputTensorAccessPattern[i];
    mapped(access, 0, 1, block_rows - 1, block_rows);
    mapped(access, 1, 0, 0, 1);
  }
  struct SoftmaxScalarParams {
    float scale;
    int32_t block_rows;
  } params{scale, static_cast<int32_t>(block_rows)};
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  return binary(output, &_binary_attn_tile_softmax_bf16_o_start, &_binary_attn_tile_softmax_bf16_o_end);
}

// mode: false = finalize, true = merge
inline GlueCodeReturn instantiate_online(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                         bool merge) {
  const unsigned expected_in = merge ? 6u : 2u;
  const unsigned expected_out = merge ? 3u : 1u;
  if (input->inputTensorNr != expected_in) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != expected_out) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t prev_unscaled = 0;
  if (merge) {
    if (input->nodeParams.nodeParamsSize == sizeof(int32_t)) {
      if (!input->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
      std::memcpy(&prev_unscaled, input->nodeParams.nodeParams, sizeof(prev_unscaled));
      if (prev_unscaled != 0 && prev_unscaled != 1) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    } else if (input->nodeParams.nodeParamsSize != 0) {
      return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
  } else if (input->nodeParams.nodeParamsSize) {
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  const auto& data = input->inputTensors[merge ? 2 : 1].geometry;
  const unsigned long width = data.maxSizes[0], rows = data.maxSizes[1];
  const unsigned long stat_rows = (rows + 63) / 64 * 64;
  if ((width != 64 && width != 128) || !rows || rows > 4096)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (data.dims == 4) {
    const unsigned long group = data.maxSizes[2], kv_heads = data.maxSizes[3];
    if (!group || group > 256 || !kv_heads || kv_heads > 256)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned row_end = static_cast<unsigned>(std::min<unsigned long>(rows, 64) - 1);
    output->indexSpaceRank = 4;
    output->indexSpaceGeometry[0] = group;
    output->indexSpaceGeometry[1] = (rows + 63) / 64;
    output->indexSpaceGeometry[2] = 1;
    output->indexSpaceGeometry[3] = kv_heads;
    for (unsigned i = 0; i < expected_in; ++i) {
      const bool is_data = merge ? (i == 2 || i == 5) : (i == 1);
      auto& access = output->inputTensorAccessPattern[i];
      if (is_data) {
        if (!shape(input->inputTensors[i], DATA_F32, {width, rows, group, kv_heads}))
          return GLUE_INCOMPATIBLE_INPUT_SIZE;
        mapped(access, 0, 2, width - 1, 0);
        mapped(access, 1, 1, row_end, 64);
        mapped(access, 2, 0, 0, 1);
        mapped(access, 3, 3, 0, 1);
      } else {
        if (!shape(input->inputTensors[i], DATA_F32, {stat_rows, group, kv_heads}))
          return GLUE_INCOMPATIBLE_INPUT_SIZE;
        mapped(access, 0, 1, 63, 64);
        mapped(access, 1, 0, 0, 1);
        mapped(access, 2, 3, 0, 1);
      }
    }
    for (unsigned i = 0; i < expected_out; ++i) {
      const bool is_data = !merge || (i == 2);
      auto& access = output->outputTensorAccessPattern[i];
      if (is_data) {
        if (!shape(input->outputTensors[i], merge ? DATA_F32 : DATA_BF16,
                   {width, rows, group, kv_heads}))
          return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        mapped(access, 0, 2, width - 1, 0);
        mapped(access, 1, 1, row_end, 64);
        mapped(access, 2, 0, 0, 1);
        mapped(access, 3, 3, 0, 1);
      } else {
        if (!shape(input->outputTensors[i], DATA_F32, {stat_rows, group, kv_heads}))
          return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        mapped(access, 0, 1, 63, 64);
        mapped(access, 1, 0, 0, 1);
        mapped(access, 2, 3, 0, 1);
      }
    }
    output->kernel.paramsNr = merge ? 1 : 0;
    if (merge) std::memcpy(output->kernel.scalarParams, &prev_unscaled, sizeof(prev_unscaled));
    if (merge)
      return binary(output, &_binary_attn_online_merge_f32_o_start, &_binary_attn_online_merge_f32_o_end);
    return binary(output, &_binary_attn_online_finalize_bf16_o_start, &_binary_attn_online_finalize_bf16_o_end);
  }
  const unsigned long heads = data.maxSizes[2];
  if (!heads || heads > 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (merge) {
    for (unsigned i : {0u, 1u, 3u, 4u})
      if (!shape(input->inputTensors[i], DATA_F32, {stat_rows, heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i : {2u, 5u})
      if (!shape(input->inputTensors[i], DATA_F32, {width, rows, heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i : {0u, 1u})
      if (!shape(input->outputTensors[i], DATA_F32, {stat_rows, heads})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (!shape(input->outputTensors[2], DATA_F32, {width, rows, heads})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  } else {
    if (!shape(input->inputTensors[0], DATA_F32, {stat_rows, heads}) ||
        !shape(input->inputTensors[1], DATA_F32, {width, rows, heads}) ||
        !shape(input->outputTensors[0], DATA_BF16, {width, rows, heads}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = (rows + 63) / 64;
  output->indexSpaceGeometry[2] = 1;
  for (unsigned i = 0; i < expected_in; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
    const bool is_data = merge ? (i == 2 || i == 5) : (i == 1);
    if (is_data) {
      mapped(access, 0, 2, width - 1, 0);
      mapped(access, 1, 1, 63, 64);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 63, 64);
      mapped(access, 1, 0, 0, 1);
    }
  }
  for (unsigned i = 0; i < expected_out; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    const bool is_data = !merge || (i == 2);
    if (is_data) {
      mapped(access, 0, 2, width - 1, 0);
      mapped(access, 1, 1, 63, 64);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 63, 64);
      mapped(access, 1, 0, 0, 1);
    }
  }
  output->kernel.paramsNr = merge ? 1 : 0;
  if (merge) std::memcpy(output->kernel.scalarParams, &prev_unscaled, sizeof(prev_unscaled));
  if (merge)
    return binary(output, &_binary_attn_online_merge_f32_o_start, &_binary_attn_online_merge_f32_o_end);
  return binary(output, &_binary_attn_online_finalize_bf16_o_start, &_binary_attn_online_finalize_bf16_o_end);
}
}  // namespace generic_attention
