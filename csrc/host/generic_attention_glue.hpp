// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>

extern "C" unsigned char _binary_attn_tile_softmax_bf16_o_start, _binary_attn_tile_softmax_bf16_o_end;
extern "C" unsigned char _binary_attn_online_merge_f32_o_start, _binary_attn_online_merge_f32_o_end;
extern "C" unsigned char _binary_attn_online_finalize_bf16_o_start, _binary_attn_online_finalize_bf16_o_end;

namespace generic_attention {
using namespace tpc_lib_api;
constexpr const char* kSoftmax = "gk_attn_tile_softmax_bf16_v1";
constexpr const char* kMerge = "gk_attn_online_merge_f32_v1";
constexpr const char* kFinalize = "gk_attn_online_finalize_bf16_v1";

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
  if (!keys || keys > 2048 || keys % 128 || !rows || rows > 256 || !heads || heads > 256 ||
      keys * rows * heads > 4UL * 1024 * 1024 ||
      !shape(input->inputTensors[0], DATA_F32, {keys, rows, heads}) ||
      !shape(input->inputTensors[1], DATA_I32, {rows}) || !shape(input->inputTensors[2], DATA_I32, {3}) ||
      !shape(input->outputTensors[0], DATA_BF16, {keys, rows, heads}) ||
      !shape(input->outputTensors[1], DATA_F32, {rows, heads}) ||
      !shape(input->outputTensors[2], DATA_F32, {rows, heads}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = rows;
  for (unsigned i : {0u, 1u}) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    mapped(access, 0, 0, keys - 1, 0);
    mapped(access, 1, 1, 0, 1);
    mapped(access, 2, 0, 0, 1);
  }
  mapped(output->inputTensorAccessPattern[1], 0, 1, 0, 1);
  output->inputTensorAccessPattern[2].allRequired = 1;
  for (unsigned i : {1u, 2u}) {
    auto& access = output->outputTensorAccessPattern[i];
    mapped(access, 0, 1, 0, 1);
    mapped(access, 1, 0, 0, 1);
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
  if ((width != 64 && width != 128) || !rows || rows > 4096 || !heads || heads > 256)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (merge) {
    for (unsigned i : {0u, 1u, 3u, 4u})
      if (!shape(input->inputTensors[i], DATA_F32, {rows, heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i : {2u, 5u})
      if (!shape(input->inputTensors[i], DATA_F32, {width, rows, heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned i : {0u, 1u})
      if (!shape(input->outputTensors[i], DATA_F32, {rows, heads})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (!shape(input->outputTensors[2], DATA_F32, {width, rows, heads})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  } else {
    if (!shape(input->inputTensors[0], DATA_F32, {rows, heads}) ||
        !shape(input->inputTensors[1], DATA_F32, {width, rows, heads}) ||
        !shape(input->outputTensors[0], DATA_BF16, {width, rows, heads}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = heads;
  output->indexSpaceGeometry[1] = rows;
  for (unsigned i = 0; i < input->inputTensorNr; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
    const bool is_data = merge ? (i == 2 || i == 5) : i == 1;
    if (is_data) {
      mapped(access, 0, 0, width - 1, 0);
      mapped(access, 1, 1, 0, 1);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 0, 1);
      mapped(access, 1, 0, 0, 1);
    }
  }
  for (unsigned i = 0; i < input->outputTensorNr; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    const bool is_data = merge ? i == 2 : true;
    if (is_data) {
      mapped(access, 0, 0, width - 1, 0);
      mapped(access, 1, 1, 0, 1);
      mapped(access, 2, 0, 0, 1);
    } else {
      mapped(access, 0, 1, 0, 1);
      mapped(access, 1, 0, 0, 1);
    }
  }
  output->kernel.paramsNr = 0;
  if (merge)
    return binary(output, &_binary_attn_online_merge_f32_o_start, &_binary_attn_online_merge_f32_o_end);
  return binary(output, &_binary_attn_online_finalize_bf16_o_start, &_binary_attn_online_finalize_bf16_o_end);
}
}  // namespace generic_attention
