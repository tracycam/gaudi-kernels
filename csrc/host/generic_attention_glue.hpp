// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tpc_kernel_lib_interface.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

extern "C" unsigned char _binary_attn_tile_softmax_bf16_o_start, _binary_attn_tile_softmax_bf16_o_end;
extern "C" unsigned char _binary_attn_pack_stats_f32_o_start, _binary_attn_pack_stats_f32_o_end;
extern "C" unsigned char _binary_attn_online_merge_f32_o_start, _binary_attn_online_merge_f32_o_end;
extern "C" unsigned char _binary_attn_online_finalize_bf16_o_start, _binary_attn_online_finalize_bf16_o_end;
extern "C" unsigned char _binary_kv_store_head_major_bf16_o_start, _binary_kv_store_head_major_bf16_o_end;
extern "C" unsigned char _binary_kv_store_head_major64_bf16_o_start, _binary_kv_store_head_major64_bf16_o_end;
extern "C" unsigned char _binary_attn_decode_reduce_bf16_o_start, _binary_attn_decode_reduce_bf16_o_end;
extern "C" unsigned char _binary_attn_direct_softmax_bf16_o_start, _binary_attn_direct_softmax_bf16_o_end;
extern "C" unsigned char _binary_attn_tile_indices_i32_o_start, _binary_attn_tile_indices_i32_o_end;

namespace generic_attention {
using namespace tpc_lib_api;
constexpr const char* kSoftmax = "gk_attn_tile_softmax_bf16_v1";
constexpr const char* kPack = "gk_attn_pack_stats_f32_v1";
constexpr const char* kMerge = "gk_attn_online_merge_f32_v1";
constexpr const char* kFinalize = "gk_attn_online_finalize_bf16_v1";
constexpr const char* kHeadStore = "gk_kv_store_head_major_bf16_v1";
constexpr const char* kDecodeReduce = "gk_attn_decode_reduce_bf16_v1";
constexpr const char* kDirectSoftmax = "gk_attn_direct_softmax_bf16_v1";
constexpr const char* kTileIndices = "gk_attn_tile_indices_i32_v1";

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

inline GlueCodeReturn instantiate_decode_reduce(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& data = input->inputTensors[2].geometry;
  const unsigned long d = data.maxSizes[0], tiles = data.maxSizes[1], heads = data.maxSizes[2];
  if ((d != 64 && d != 128) || !tiles || tiles > 16 || !heads || heads > 256 ||
      !shape(input->inputTensors[0], DATA_F32, {64, tiles, heads}) ||
      !shape(input->inputTensors[1], DATA_F32, {64, tiles, heads}) ||
      !shape(input->inputTensors[2], DATA_F32, {d, tiles, heads}) ||
      !shape(input->outputTensors[0], DATA_BF16, {d, heads, 1}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = output->indexSpaceGeometry[2] = 1;
  for (unsigned i = 0; i < 3; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
    mapped(access, 0, 1, (i == 2 ? d : 64) - 1, 0);
    mapped(access, 1, 2, tiles - 1, 0);
    mapped(access, 2, 0, 0, 1);
  }
  auto& result = output->outputTensorAccessPattern[0];
  mapped(result, 0, 1, d - 1, 0);
  mapped(result, 1, 0, 0, 1);
  mapped(result, 2, 2, 0, 0);
  output->kernel.paramsNr = 0;
  return binary(output, &_binary_attn_decode_reduce_bf16_o_start, &_binary_attn_decode_reduce_bf16_o_end);
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
  const unsigned long keys = data.maxSizes[0], rows = data.maxSizes[1], heads = data.maxSizes[2];
  if (!keys || keys > 512 || keys % 128 || !rows || rows > 256 || !heads || heads > 256 ||
      keys * rows * heads > 4UL * 1024 * 1024 ||
      !shape(input->inputTensors[0], DATA_F32, {keys, rows, heads}) ||
      !shape(input->inputTensors[1], DATA_I32, {rows}) ||
      !shape(input->inputTensors[2], DATA_I32, {3}) ||
      !shape(input->outputTensors[0], DATA_BF16, {keys, rows, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = rows;
  output->indexSpaceGeometry[2] = 1;
  for (auto* access : {&output->inputTensorAccessPattern[0], &output->outputTensorAccessPattern[0]}) {
    mapped(*access, 0, 2, keys - 1, 0);
    mapped(*access, 1, 1, 0, 1);
    mapped(*access, 2, 0, 0, 1);
  }
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->inputTensorAccessPattern[2].allRequired = 1;
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &scale, sizeof(scale));
  return binary(output, &_binary_attn_direct_softmax_bf16_o_start,
                &_binary_attn_direct_softmax_bf16_o_end);
}

inline GlueCodeReturn instantiate_pack(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& data=input->inputTensors[0].geometry;
  const unsigned long rows=data.maxSizes[1], heads=data.maxSizes[2], padded=(rows+63)/64*64;
  if (!rows || rows>256 || !heads || heads>256 ||
      !shape(input->inputTensors[0],DATA_F32,{64,rows,heads}) ||
      !shape(input->inputTensors[1],DATA_F32,{64,rows,heads}) ||
      !shape(input->outputTensors[0],DATA_F32,{padded,heads}) ||
      !shape(input->outputTensors[1],DATA_F32,{padded,heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank=3;
  output->indexSpaceGeometry[0]=heads; output->indexSpaceGeometry[1]=(rows+63)/64;
  output->indexSpaceGeometry[2]=1;
  for (unsigned i=0; i<2; ++i) {
    auto& a=output->inputTensorAccessPattern[i];
    mapped(a,0,2,63,0); mapped(a,1,1,63,64); mapped(a,2,0,0,1);
    auto& b=output->outputTensorAccessPattern[i];
    mapped(b,0,1,63,64); mapped(b,1,0,0,1);
  }
  output->kernel.paramsNr=0;
  return binary(output,&_binary_attn_pack_stats_f32_o_start,&_binary_attn_pack_stats_f32_o_end);
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
  const unsigned long keys = score.maxSizes[0], rows = score.maxSizes[1], heads = score.maxSizes[2];
  // Keep each score tile bounded. Full prompts are dispatched as query tiles.
  if (!keys || keys > 512 || keys % 128 || !rows || rows > 256 || !heads || heads > 256 ||
      keys * rows * heads > 4UL * 1024 * 1024 ||
      !shape(input->inputTensors[0], DATA_F32, {keys, rows, heads}) ||
      !shape(input->inputTensors[1], DATA_I32, {rows}) || !shape(input->inputTensors[2], DATA_I32, {3}) ||
      !shape(input->outputTensors[0], DATA_BF16, {keys, rows, heads}) ||
      !shape(input->outputTensors[1], DATA_F32, {64, rows, heads}) ||
      !shape(input->outputTensors[2], DATA_F32, {64, rows, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = rows;
  output->indexSpaceGeometry[2] = 1;
  for (unsigned i : {0u, 1u}) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    mapped(access, 0, 2, keys - 1, 0);
    mapped(access, 1, 1, 0, 1);
    mapped(access, 2, 0, 0, 1);
  }
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->inputTensorAccessPattern[2].allRequired = 1;
  for (unsigned i : {1u, 2u}) {
    auto& access = output->outputTensorAccessPattern[i];
    mapped(access, 0, 2, 63, 0);
    mapped(access, 1, 1, 0, 1);
    mapped(access, 2, 0, 0, 1);
  }
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &scale, sizeof(scale));
  return binary(output, &_binary_attn_tile_softmax_bf16_o_start, &_binary_attn_tile_softmax_bf16_o_end);
}

inline GlueCodeReturn instantiate_online(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                         bool merge) {
  if (input->inputTensorNr != (merge ? 6u : 2u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != (merge ? 3u : 1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& data = input->inputTensors[merge ? 2 : 1].geometry;
  const unsigned long width = data.maxSizes[0], rows = data.maxSizes[1], heads = data.maxSizes[2];
  const unsigned long stat_rows = (rows + 63) / 64 * 64;
  if ((width != 64 && width != 128) || !rows || rows > 4096 || !heads || heads > 256)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
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
  for (unsigned i = 0; i < input->inputTensorNr; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
    const bool is_data = merge ? (i == 2 || i == 5) : i == 1;
    if (is_data) {
      mapped(access, 0, 2, width - 1, 0);
      mapped(access, 1, 1, 63, 64);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 63, 64);
      mapped(access, 1, 0, 0, 1);
    }
  }
  for (unsigned i = 0; i < input->outputTensorNr; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    const bool is_data = merge ? i == 2 : true;
    if (is_data) {
      mapped(access, 0, 2, width - 1, 0);
      mapped(access, 1, 1, 63, 64);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 63, 64);
      mapped(access, 1, 0, 0, 1);
    }
  }
  output->kernel.paramsNr = 0;
  if (merge)
    return binary(output, &_binary_attn_online_merge_f32_o_start, &_binary_attn_online_merge_f32_o_end);
  return binary(output, &_binary_attn_online_finalize_bf16_o_start, &_binary_attn_online_finalize_bf16_o_end);
}
}  // namespace generic_attention
