// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include "generic_attention_glue.hpp"
#include "generic_fp8_glue.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_add_bf16_o_start, _binary_add_bf16_o_end;
extern "C" unsigned char _binary_fused_add_rms_norm_bf16_o_start, _binary_fused_add_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_rms_norm_bf16_o_start, _binary_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_silu_and_mul_bf16_o_start, _binary_silu_and_mul_bf16_o_end;
extern "C" unsigned char _binary_rotary_bf16_o_start, _binary_rotary_bf16_o_end;
extern "C" unsigned char _binary_rotary_pos_bf16_o_start, _binary_rotary_pos_bf16_o_end;
extern "C" unsigned char _binary_ewise_unary_bf16_o_start, _binary_ewise_unary_bf16_o_end;
extern "C" unsigned char _binary_ewise_unary_f32_o_start, _binary_ewise_unary_f32_o_end;
extern "C" unsigned char _binary_gelu_and_mul_bf16_o_start, _binary_gelu_and_mul_bf16_o_end;
extern "C" unsigned char _binary_sample_vocab_bf16_o_start, _binary_sample_vocab_bf16_o_end;
extern "C" unsigned char _binary_sample_vocab_f32_o_start, _binary_sample_vocab_f32_o_end;
extern "C" unsigned char _binary_apply_vocab_bitmask_bf16_o_start, _binary_apply_vocab_bitmask_bf16_o_end;
extern "C" unsigned char _binary_apply_vocab_bitmask_f32_o_start, _binary_apply_vocab_bitmask_f32_o_end;
extern "C" unsigned char _binary_rank_sum_bf16_o_start, _binary_rank_sum_bf16_o_end;
extern "C" unsigned char _binary_rank_sum_f32_o_start, _binary_rank_sum_f32_o_end;
extern "C" unsigned char _binary_paged_metadata_i32_o_start, _binary_paged_metadata_i32_o_end;
extern "C" unsigned char _binary_token_indices_i32_o_start, _binary_token_indices_i32_o_end;
extern "C" unsigned char _binary_paged_alloc_i32_o_start, _binary_paged_alloc_i32_o_end;
extern "C" unsigned char _binary_paged_free_i32_o_start, _binary_paged_free_i32_o_end;
extern "C" unsigned char _binary_cast_f32_to_bf16_o_start, _binary_cast_f32_to_bf16_o_end;
extern "C" unsigned char _binary_index_put_i64_o_start, _binary_index_put_i64_o_end;
extern "C" unsigned char _binary_argmax_parts_bf16_o_start, _binary_argmax_parts_bf16_o_end;
extern "C" unsigned char _binary_argmax_parts_f32_o_start, _binary_argmax_parts_f32_o_end;
extern "C" unsigned char _binary_argmax_finish_i32_o_start, _binary_argmax_finish_i32_o_end;

namespace {
constexpr const char* kAdd = "gk_add_bf16_v1";
constexpr const char* kFusedNorm = "gk_fused_add_rms_norm_bf16_v1";
constexpr const char* kNorm = "gk_rms_norm_bf16_v1";
constexpr const char* kSilu = "gk_silu_and_mul_bf16_v1";
constexpr const char* kRotary = "gk_rotary_bf16_v1";
constexpr const char* kRotaryPos = "gk_rotary_pos_bf16_v1";
constexpr const char* kEwiseUnaryBf16 = "gk_ewise_unary_bf16_v1";
constexpr const char* kEwiseUnaryF32 = "gk_ewise_unary_f32_v1";
constexpr const char* kGeluAndMul = "gk_gelu_and_mul_bf16_v1";
constexpr const char* kSampleVocabBf16 = "gk_sample_vocab_bf16_v1";
constexpr const char* kSampleVocabF32 = "gk_sample_vocab_f32_v1";
constexpr const char* kApplyVocabBitmaskBf16 = "gk_apply_vocab_bitmask_bf16_v1";
constexpr const char* kApplyVocabBitmaskF32 = "gk_apply_vocab_bitmask_f32_v1";
constexpr const char* kRankSumBf16 = "gk_rank_sum_bf16_v1";
constexpr const char* kRankSumF32 = "gk_rank_sum_f32_v1";
constexpr const char* kPagedMetadata = "gk_paged_metadata_i32_v1";
constexpr const char* kTokenIndices = "gk_token_indices_i32_v1";
constexpr const char* kSchedulerMetadata = "gk_scheduler_metadata_i32_v1";
constexpr const char* kSchedulerTokens = "gk_scheduler_tokens_i32_v1";
constexpr const char* kCastF32ToBf16 = "gk_cast_f32_to_bf16_v1";
constexpr const char* kIndexPutI64 = "gk_index_put_i64_v1";
constexpr const char* kPagedAlloc = "gk_paged_alloc_i32_v1";
constexpr const char* kPagedFree = "gk_paged_free_i32_v1";
constexpr const char* kArgmaxBf16 = "gk_argmax_parts_bf16_v1";
constexpr const char* kArgmaxF32 = "gk_argmax_parts_f32_v1";
constexpr const char* kArgmaxFinish = "gk_argmax_finish_i32_v1";

enum class ArgmaxKind { part_f32, part_bf16, finish_i32 };

template <ArgmaxKind Kind>
GlueCodeReturn instantiate_argmax(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  constexpr bool finish = (Kind == ArgmaxKind::finish_i32);
  if (input->inputTensorNr != (finish ? 3u : 1u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != (finish ? 1u : 3u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != (finish ? sizeof(int) : 2 * sizeof(int)) ||
      !input->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  int params[2]{};
  std::memcpy(params, input->nodeParams.nodeParams, input->nodeParams.nodeParamsSize);
  const auto& source = input->inputTensors[0].geometry;
  const auto rows = finish ? source.maxSizes[2] : source.maxSizes[1];
  const auto parts = finish ? source.maxSizes[1] : input->outputTensors[0].geometry.maxSizes[1];
  if (!rows || rows > INT_MAX || !parts || parts > 24) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if constexpr (finish) {
    if (params[0] != static_cast<int>(parts) ||
        !generic_attention::shape(input->inputTensors[0], DATA_F32, {64, parts, rows}) ||
        !generic_attention::shape(input->inputTensors[1], DATA_I32, {64, parts, rows}) ||
        !generic_attention::shape(input->inputTensors[2], DATA_I32, {64, parts, rows}) ||
        !generic_attention::shape(input->outputTensors[0], DATA_I32, {1, rows}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    output->indexSpaceRank = 2;
    output->indexSpaceGeometry[0] = rows;
    output->indexSpaceGeometry[1] = 1;
    for (unsigned i = 0; i < 3; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
    generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 1, 0, 0);
    generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 0, 0, 1);
    output->kernel.paramsNr = 1;
    std::memcpy(output->kernel.scalarParams, params, sizeof(int));
    return generic_attention::binary(output, &_binary_argmax_finish_i32_o_start, &_binary_argmax_finish_i32_o_end);
  } else {
    const auto width = source.maxSizes[0];
    const auto expected_chunk = ((width + 127) / 128 + parts - 1) / parts;
    constexpr auto dtype = (Kind == ArgmaxKind::part_bf16) ? DATA_BF16 : DATA_F32;
    if (!width || width > INT_MAX || params[0] != static_cast<int>(width) ||
        params[1] != static_cast<int>(expected_chunk) ||
        !generic_attention::shape(input->inputTensors[0], dtype, {width, rows}) ||
        !generic_attention::shape(input->outputTensors[0], DATA_F32, {64, parts, rows}) ||
        !generic_attention::shape(input->outputTensors[1], DATA_I32, {64, parts, rows}) ||
        !generic_attention::shape(input->outputTensors[2], DATA_I32, {64, parts, rows}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    output->indexSpaceRank = 3;
    output->indexSpaceGeometry[0] = parts;
    output->indexSpaceGeometry[1] = rows;
    output->indexSpaceGeometry[2] = 1;
    generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 0,
                              static_cast<unsigned>(expected_chunk * 128 - 1),
                              static_cast<unsigned>(expected_chunk * 128));
    generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, 0, 1);
    for (unsigned i = 0; i < 3; ++i) {
      auto& access = output->outputTensorAccessPattern[i];
      generic_attention::mapped(access, 0, 2, 63, 0);
      generic_attention::mapped(access, 1, 0, 0, 1);
      generic_attention::mapped(access, 2, 1, 0, 1);
    }
    output->kernel.paramsNr = 2;
    std::memcpy(output->kernel.scalarParams, params, 2 * sizeof(int));
    if constexpr (Kind == ArgmaxKind::part_bf16) {
      return generic_attention::binary(output, &_binary_argmax_parts_bf16_o_start,
                                       &_binary_argmax_parts_bf16_o_end);
    } else {
      return generic_attention::binary(output, &_binary_argmax_parts_f32_o_start,
                                       &_binary_argmax_parts_f32_o_end);
    }
  }
}

GlueCodeReturn instantiate_norm(HabanaKernelParams* input, HabanaKernelInstantiation* output, bool fused) {
  const unsigned inputs = fused ? 3 : 2;
  const unsigned outputs = fused ? 2 : 1;
  const unsigned gamma = inputs - 1;
  if (input->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct NormParams {
    float epsilon = 0.0f;
    float weight_offset = 0.0f;
  } params{};
  if (!input->nodeParams.nodeParams ||
      (input->nodeParams.nodeParamsSize != sizeof(float) &&
       input->nodeParams.nodeParamsSize != sizeof(NormParams)))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, input->nodeParams.nodeParamsSize);
  if (!std::isfinite(params.epsilon) || params.epsilon <= 0.0f ||
      !std::isfinite(params.weight_offset))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& shape = input->inputTensors[0].geometry;
  if (shape.dims != 2 || !shape.maxSizes[0] || shape.maxSizes[0] > 8192 ||
      !shape.maxSizes[1] || shape.maxSizes[1] > INT_MAX)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < inputs + outputs; ++i) {
    const auto& tensor = i < inputs ? input->inputTensors[i].geometry : input->outputTensors[i - inputs].geometry;
    const auto rank = i == gamma ? 1u : 2u;
    if (tensor.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (tensor.dims != rank || tensor.maxSizes[0] != shape.maxSizes[0] ||
        (rank == 2 && tensor.maxSizes[1] != shape.maxSizes[1]) ||
        tensor.minSizes[0] != tensor.maxSizes[0] ||
        (rank == 2 && tensor.minSizes[1] != tensor.maxSizes[1]))
      return i < inputs ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = shape.maxSizes[1];
  output->indexSpaceGeometry[1] = 1;
  for (unsigned i = 0; i < inputs + outputs; ++i) {
    auto& access = i < inputs ? output->inputTensorAccessPattern[i] : output->outputTensorAccessPattern[i - inputs];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 1;
    access.mapping[0].a = 0;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = shape.maxSizes[0] - 1;
    if (i != gamma) {
      access.mapping[1].indexSpaceDim = 0;
      access.mapping[1].a = 1;
      access.mapping[1].start_b = 0;
      access.mapping[1].end_b = 0;
    }
  }
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  auto* start = fused ? &_binary_fused_add_rms_norm_bf16_o_start : &_binary_rms_norm_bf16_o_start;
  auto* end = fused ? &_binary_fused_add_rms_norm_bf16_o_end : &_binary_rms_norm_bf16_o_end;
  const unsigned size = static_cast<unsigned>(end - start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_silu(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& source = input->inputTensors[0].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (source.dataType != DATA_BF16 || result.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (source.dims != 2 || !source.maxSizes[0] || source.maxSizes[0] > INT_MAX ||
      source.maxSizes[0] % 2 || !source.maxSizes[1] || source.maxSizes[1] > INT_MAX ||
      source.minSizes[0] != source.maxSizes[0] || source.minSizes[1] != source.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const auto width = source.maxSizes[0] / 2;
  if (result.dims != 2 || result.maxSizes[0] != width || result.maxSizes[1] != source.maxSizes[1] ||
      result.minSizes[0] != result.maxSizes[0] || result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  constexpr unsigned kColStep = 512;
  constexpr unsigned kRowTile = 8;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (width + kColStep - 1) / kColStep;
  output->indexSpaceGeometry[1] = (source.maxSizes[1] + kRowTile - 1) / kRowTile;
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = kColStep;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = (i ? 0 : width) + kColStep - 1;
    access.mapping[1].indexSpaceDim = 1;
    access.mapping[1].a = kRowTile;
    access.mapping[1].start_b = 0;
    access.mapping[1].end_b = kRowTile - 1;
  }
  output->kernel.paramsNr = 0;
  const unsigned size = static_cast<unsigned>(&_binary_silu_and_mul_bf16_o_end -
                                               &_binary_silu_and_mul_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_silu_and_mul_bf16_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_add_bf16(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& shape = input->inputTensors[0].geometry;
  if (shape.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (shape.dims != 2 || !shape.maxSizes[0] || shape.maxSizes[0] > INT_MAX ||
      !shape.maxSizes[1] || shape.maxSizes[1] > INT_MAX)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 3; ++i) {
    const auto& tensor = i < 2 ? input->inputTensors[i].geometry : input->outputTensors[0].geometry;
    if (tensor.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (tensor.dims != 2 || tensor.maxSizes[0] != shape.maxSizes[0] ||
        tensor.maxSizes[1] != shape.maxSizes[1] || tensor.minSizes[0] != tensor.maxSizes[0] ||
        tensor.minSizes[1] != tensor.maxSizes[1])
      return i < 2 ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (shape.maxSizes[0] + 127) / 128;
  output->indexSpaceGeometry[1] = shape.maxSizes[1];
  for (unsigned i = 0; i < 3; ++i) {
    auto& access = i < 2 ? output->inputTensorAccessPattern[i] : output->outputTensorAccessPattern[0];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = 128;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = 127;
    access.mapping[1].indexSpaceDim = 1;
    access.mapping[1].a = 1;
    access.mapping[1].start_b = 0;
    access.mapping[1].end_b = 0;
  }
  output->kernel.paramsNr = 0;
  return generic_attention::binary(output, &_binary_add_bf16_o_start, &_binary_add_bf16_o_end);
}

GlueCodeReturn instantiate_rotary(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params { int32_t head_dim; int32_t rotary_dim; int32_t is_neox; } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.head_dim <= 0 || params.rotary_dim <= 0 || params.rotary_dim > params.head_dim ||
      params.rotary_dim % 2 || (params.is_neox != 0 && params.is_neox != 1))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& query = input->inputTensors[0].geometry;
  const auto& key = input->inputTensors[1].geometry;
  const auto& positions = input->inputTensors[2].geometry;
  const auto& cache = input->inputTensors[3].geometry;
  const auto& metadata = input->inputTensors[4].geometry;
  if (query.dataType != DATA_BF16 || key.dataType != DATA_BF16 || cache.dataType != DATA_BF16 ||
      positions.dataType != DATA_I32 || metadata.dataType != DATA_I32 ||
      input->outputTensors[0].geometry.dataType != DATA_BF16 ||
      input->outputTensors[1].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  for (const auto* tensor : {&query, &key}) {
    if (tensor->dims != 2 || !tensor->maxSizes[0] || tensor->maxSizes[0] > INT_MAX ||
        tensor->maxSizes[0] % params.head_dim || !tensor->maxSizes[1] || tensor->maxSizes[1] > 8192 ||
        tensor->minSizes[0] != tensor->maxSizes[0] || tensor->minSizes[1] != tensor->maxSizes[1])
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  if (key.maxSizes[1] != query.maxSizes[1] || positions.dims != 1 ||
      positions.maxSizes[0] != query.maxSizes[1] || positions.minSizes[0] != positions.maxSizes[0] ||
      metadata.dims != 1 || metadata.maxSizes[0] != 1 || metadata.minSizes[0] != 1 ||
      cache.dims != 2 || cache.maxSizes[0] != static_cast<unsigned>(params.rotary_dim) ||
      cache.minSizes[0] != cache.maxSizes[0] || !cache.maxSizes[1] || cache.maxSizes[1] > INT_MAX ||
      !cache.minSizes[1] || cache.minSizes[1] > cache.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i) {
    const auto& source = input->inputTensors[i].geometry;
    const auto& result = input->outputTensors[i].geometry;
    if (result.dims != 2 || result.maxSizes[0] != source.maxSizes[0] ||
        result.maxSizes[1] != source.maxSizes[1] || result.minSizes[0] != result.maxSizes[0] ||
        result.minSizes[1] != result.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = std::max(query.maxSizes[0], key.maxSizes[0]) / params.head_dim;
  output->indexSpaceGeometry[1] = query.maxSizes[1];
  for (unsigned i = 0; i < 5; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < 2; ++i) output->outputTensorAccessPattern[i].allRequired = 1;
  output->kernel.paramsNr = 3;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  const unsigned size = static_cast<unsigned>(&_binary_rotary_bf16_o_end - &_binary_rotary_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_rotary_bf16_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_rotary_pos(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params { int32_t head_dim; int32_t rotary_dim; int32_t is_neox; } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.head_dim <= 0 || params.rotary_dim <= 0 || params.rotary_dim > params.head_dim ||
      params.rotary_dim % 2 || (params.is_neox != 0 && params.is_neox != 1))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& query = input->inputTensors[0].geometry;
  const auto& key = input->inputTensors[1].geometry;
  const auto& positions = input->inputTensors[2].geometry;
  const auto& cache = input->inputTensors[3].geometry;
  if (query.dataType != DATA_BF16 || key.dataType != DATA_BF16 || cache.dataType != DATA_BF16 ||
      positions.dataType != DATA_I32 ||
      input->outputTensors[0].geometry.dataType != DATA_BF16 ||
      input->outputTensors[1].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  for (const auto* tensor : {&query, &key}) {
    if (tensor->dims != 2 || !tensor->maxSizes[0] || tensor->maxSizes[0] > INT_MAX ||
        tensor->maxSizes[0] % params.head_dim || !tensor->maxSizes[1] || tensor->maxSizes[1] > INT_MAX ||
        tensor->minSizes[0] != tensor->maxSizes[0] || tensor->minSizes[1] != tensor->maxSizes[1])
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  if (key.maxSizes[1] != query.maxSizes[1] || positions.dims != 1 ||
      positions.maxSizes[0] != query.maxSizes[1] || positions.minSizes[0] != positions.maxSizes[0] ||
      cache.dims != 2 || cache.maxSizes[0] != static_cast<unsigned>(params.rotary_dim) ||
      cache.minSizes[0] != cache.maxSizes[0] || !cache.maxSizes[1] || cache.maxSizes[1] > INT_MAX ||
      !cache.minSizes[1] || cache.minSizes[1] > cache.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i) {
    const auto& source = input->inputTensors[i].geometry;
    const auto& result = input->outputTensors[i].geometry;
    if (result.dims != 2 || result.maxSizes[0] != source.maxSizes[0] ||
        result.maxSizes[1] != source.maxSizes[1] || result.minSizes[0] != result.maxSizes[0] ||
        result.minSizes[1] != result.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  const unsigned long total_heads =
      query.maxSizes[0] / params.head_dim + key.maxSizes[0] / params.head_dim;
  const unsigned long rows = query.maxSizes[1];
  const unsigned long cores = (rows >= 24 || total_heads >= 24)
                                  ? 24UL
                                  : std::min<unsigned long>(24UL, rows * total_heads);
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = std::max<unsigned long>(1UL, cores);
  for (unsigned i = 0; i < 4; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < 2; ++i) output->outputTensorAccessPattern[i].allRequired = 1;
  output->kernel.paramsNr = 3;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  return generic_attention::binary(output, &_binary_rotary_pos_bf16_o_start,
                                   &_binary_rotary_pos_bf16_o_end);
}

GlueCodeReturn instantiate_ewise_unary(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                       bool is_bf16) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params {
    int32_t op_code;
    float alpha;
    float beta;
  } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.op_code < 0 || params.op_code > 11 ||
      std::isnan(params.alpha) || std::isnan(params.beta) ||
      (params.op_code == 6 && (!std::isfinite(params.alpha) || params.alpha == 0.0f)))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto dtype = is_bf16 ? DATA_BF16 : DATA_F32;
  const auto& src = input->inputTensors[0].geometry;
  const auto& dst = input->outputTensors[0].geometry;
  if (src.dataType != dtype || dst.dataType != dtype) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (src.dims != 2 || !src.maxSizes[0] || src.maxSizes[0] > INT_MAX ||
      !src.maxSizes[1] || src.maxSizes[1] > INT_MAX ||
      src.minSizes[0] != src.maxSizes[0] || src.minSizes[1] != src.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (dst.dims != 2 || dst.maxSizes[0] != src.maxSizes[0] || dst.maxSizes[1] != src.maxSizes[1] ||
      dst.minSizes[0] != dst.maxSizes[0] || dst.minSizes[1] != dst.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  const unsigned long step = is_bf16 ? 128UL : 64UL;
  const unsigned long chunks = (src.maxSizes[0] + step - 1) / step;
  const unsigned long cores = src.maxSizes[1] >= 24
                                  ? 24UL
                                  : std::min<unsigned long>(24UL, src.maxSizes[1] * chunks);
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = std::max<unsigned long>(1UL, cores);
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 3;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  return is_bf16
      ? generic_attention::binary(output, &_binary_ewise_unary_bf16_o_start,
                                  &_binary_ewise_unary_bf16_o_end)
      : generic_attention::binary(output, &_binary_ewise_unary_f32_o_start,
                                  &_binary_ewise_unary_f32_o_end);
}

GlueCodeReturn instantiate_gelu_and_mul(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t approximate = 0;
  if (input->nodeParams.nodeParamsSize == sizeof(approximate)) {
    if (!input->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    std::memcpy(&approximate, input->nodeParams.nodeParams, sizeof(approximate));
    if (approximate != 0 && approximate != 1) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  } else if (input->nodeParams.nodeParamsSize != 0) {
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  const auto& source = input->inputTensors[0].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (source.dataType != DATA_BF16 || result.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (source.dims != 2 || !source.maxSizes[0] || source.maxSizes[0] > INT_MAX ||
      source.maxSizes[0] % 2 || !source.maxSizes[1] || source.maxSizes[1] > INT_MAX ||
      source.minSizes[0] != source.maxSizes[0] || source.minSizes[1] != source.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const auto width = source.maxSizes[0] / 2;
  if (result.dims != 2 || result.maxSizes[0] != width || result.maxSizes[1] != source.maxSizes[1] ||
      result.minSizes[0] != result.maxSizes[0] || result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  const unsigned long chunks = (width + 127) / 128;
  const unsigned long cores = source.maxSizes[1] >= 24
                                  ? 24UL
                                  : std::min<unsigned long>(24UL, source.maxSizes[1] * chunks);
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = std::max<unsigned long>(1UL, cores);
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &approximate, sizeof(approximate));
  return generic_attention::binary(output, &_binary_gelu_and_mul_bf16_o_start,
                                   &_binary_gelu_and_mul_bf16_o_end);
}

GlueCodeReturn instantiate_sample_vocab(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                        bool is_bf16) {
  if (input->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params {
    int32_t vocab_size;
    int32_t nan_to_neg_inf;
  } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.vocab_size <= 0 || (params.nan_to_neg_inf != 0 && params.nan_to_neg_inf != 1))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& logits = input->inputTensors[0].geometry;
  const auto& out = input->outputTensors[0].geometry;
  const auto logits_type = is_bf16 ? DATA_BF16 : DATA_F32;
  if (logits.dataType != logits_type ||
      input->inputTensors[1].geometry.dataType != DATA_F32 ||
      input->inputTensors[2].geometry.dataType != DATA_F32 ||
      input->inputTensors[3].geometry.dataType != DATA_I32 ||
      input->inputTensors[4].geometry.dataType != DATA_F32 ||
      input->inputTensors[5].geometry.dataType != DATA_I32 ||
      out.dataType != DATA_I32)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (logits.dims != 2 || !logits.maxSizes[0] || logits.maxSizes[0] > INT_MAX ||
      !logits.maxSizes[1] || logits.maxSizes[1] > INT_MAX ||
      logits.minSizes[0] != logits.maxSizes[0] || logits.minSizes[1] != logits.maxSizes[1] ||
      static_cast<unsigned long>(params.vocab_size) > logits.maxSizes[0])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const auto rows = logits.maxSizes[1];
  for (unsigned i = 1; i <= 4; ++i) {
    if (!generic_attention::shape(input->inputTensors[i],
                                  i == 3 ? DATA_I32 : DATA_F32, {rows}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  if (!generic_attention::shape(input->inputTensors[5], DATA_I32, {2, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (!generic_attention::shape(input->outputTensors[0], DATA_I32, {rows}))
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = std::min<unsigned long>(24UL, rows);
  for (unsigned i = 0; i < 6; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  return is_bf16
      ? generic_attention::binary(output, &_binary_sample_vocab_bf16_o_start,
                                  &_binary_sample_vocab_bf16_o_end)
      : generic_attention::binary(output, &_binary_sample_vocab_f32_o_start,
                                  &_binary_sample_vocab_f32_o_end);
}

GlueCodeReturn instantiate_apply_vocab_bitmask(HabanaKernelParams* input,
                                               HabanaKernelInstantiation* output,
                                               bool is_bf16) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params {
    int32_t active_cols;
    int32_t mask_tail;
  } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.active_cols < 0 || (params.mask_tail != 0 && params.mask_tail != 1))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;

  const auto& logits = input->inputTensors[0].geometry;
  const auto& bitmask = input->inputTensors[1].geometry;
  const auto& out = input->outputTensors[0].geometry;
  const auto logits_type = is_bf16 ? DATA_BF16 : DATA_F32;
  if (logits.dataType != logits_type || bitmask.dataType != DATA_I32 || out.dataType != logits_type)
    return GLUE_INCOMPATIBLE_DATA_TYPE;

  if (logits.dims != 2 || !logits.maxSizes[0] || logits.maxSizes[0] > INT_MAX ||
      !logits.maxSizes[1] || logits.maxSizes[1] > INT_MAX ||
      logits.minSizes[0] != logits.maxSizes[0] || logits.minSizes[1] != logits.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const auto vocab_cols = logits.maxSizes[0];
  const auto rows = logits.maxSizes[1];
  if (bitmask.dims != 2 || !bitmask.maxSizes[0] || bitmask.maxSizes[0] > INT_MAX ||
      bitmask.maxSizes[1] != rows || bitmask.minSizes[0] != bitmask.maxSizes[0] ||
      bitmask.minSizes[1] != bitmask.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (static_cast<uint64_t>(params.active_cols) > vocab_cols ||
      static_cast<uint64_t>(params.active_cols) > bitmask.maxSizes[0] * 32ULL)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (out.dims != 2 || out.maxSizes[0] != vocab_cols || out.maxSizes[1] != rows ||
      out.minSizes[0] != vocab_cols || out.minSizes[1] != rows)
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;

  const unsigned long step = is_bf16 ? 128UL : 64UL;
  const unsigned long chunks = (vocab_cols + step - 1) / step;
  const unsigned long total = rows * chunks;
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = std::max<unsigned long>(1UL, std::min<unsigned long>(24UL, total));
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  return is_bf16
      ? generic_attention::binary(output, &_binary_apply_vocab_bitmask_bf16_o_start,
                                  &_binary_apply_vocab_bitmask_bf16_o_end)
      : generic_attention::binary(output, &_binary_apply_vocab_bitmask_f32_o_start,
                                  &_binary_apply_vocab_bitmask_f32_o_end);
}

GlueCodeReturn instantiate_rank_sum(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                    bool is_bf16) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t world_size = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(world_size) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&world_size, input->nodeParams.nodeParams, sizeof(world_size));
  if (world_size < 2 || world_size > 8) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;

  const auto& gathered = input->inputTensors[0].geometry;
  const auto& dst = input->outputTensors[0].geometry;
  const auto expected = is_bf16 ? DATA_BF16 : DATA_F32;
  if (gathered.dataType != expected || dst.dataType != expected) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (dst.dims != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  if (!dst.maxSizes[0] || !dst.minSizes[0] || dst.minSizes[0] > dst.maxSizes[0] ||
      dst.maxSizes[0] > static_cast<uint64_t>(INT_MAX) / static_cast<uint64_t>(world_size))
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  if (gathered.dims != 1 ||
      gathered.maxSizes[0] != dst.maxSizes[0] * static_cast<uint64_t>(world_size) ||
      gathered.minSizes[0] != dst.minSizes[0] * static_cast<uint64_t>(world_size))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;

  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = 24;
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &world_size, sizeof(world_size));
  return is_bf16
      ? generic_attention::binary(output, &_binary_rank_sum_bf16_o_start, &_binary_rank_sum_bf16_o_end)
      : generic_attention::binary(output, &_binary_rank_sum_f32_o_start, &_binary_rank_sum_f32_o_end);
}

GlueCodeReturn instantiate_paged_metadata(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                          bool scheduler_owned) {
  if (input->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 4) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t slots = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(slots) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&slots, input->nodeParams.nodeParams, sizeof(slots));
  if (slots <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  for (unsigned i = 0; i < 10; ++i) {
    const auto& tensor = i < 6 ? input->inputTensors[i].geometry : input->outputTensors[i - 6].geometry;
    if (tensor.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
  }
  const auto& locations = input->inputTensors[0].geometry;
  const auto bucket = locations.maxSizes[1];
  if (locations.dims != 2 || !bucket || bucket > 8192) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i : {0u, 1u, 2u}) {
    const auto& tensor = input->inputTensors[i].geometry;
    if (tensor.dims != 2 || (tensor.maxSizes[0] != 1 && tensor.maxSizes[0] != 2) ||
        tensor.minSizes[0] != tensor.maxSizes[0] || tensor.maxSizes[1] != bucket || tensor.minSizes[1] != bucket)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  const auto& mapping = input->inputTensors[3].geometry;
  const auto& header = input->inputTensors[4].geometry;
  const auto& info = input->inputTensors[5].geometry;
  if (mapping.dims != 2 || !mapping.maxSizes[0] || mapping.maxSizes[0] > INT_MAX ||
      !mapping.maxSizes[1] || mapping.maxSizes[1] > INT_MAX ||
      mapping.minSizes[0] != mapping.maxSizes[0] || mapping.minSizes[1] != mapping.maxSizes[1] ||
      header.dims != 1 || header.maxSizes[0] != 1 || header.minSizes[0] != 1 ||
      info.dims != 1 || info.maxSizes[0] != 1 + 4 * bucket || info.minSizes[0] != info.maxSizes[0])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 4; ++i) {
    const auto& result = input->outputTensors[i].geometry;
    if (result.dims != 1 || result.maxSizes[0] != bucket || result.minSizes[0] != bucket)
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = bucket;
  for (unsigned i = 0; i < 6; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < 4; ++i) output->outputTensorAccessPattern[i].allRequired = 1;
  const int32_t params[] = {slots, scheduler_owned ? 1 : 0};
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, params, sizeof(params));
  const unsigned size = static_cast<unsigned>(&_binary_paged_metadata_i32_o_end - &_binary_paged_metadata_i32_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_paged_metadata_i32_o_start, size);
  return GLUE_SUCCESS;
}
GlueCodeReturn instantiate_token_indices(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                         bool scheduler_owned) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t vocabulary = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(vocabulary) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&vocabulary, input->nodeParams.nodeParams, sizeof(vocabulary));
  if (vocabulary <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& tokens = input->inputTensors[0].geometry;
  const auto& header = input->inputTensors[1].geometry;
  const auto bucket = tokens.maxSizes[1];
  if (tokens.dataType != DATA_I32 || header.dataType != DATA_I32 || tokens.dims != 2 ||
      (tokens.maxSizes[0] != 1 && tokens.maxSizes[0] != 2) ||
      tokens.minSizes[0] != tokens.maxSizes[0] || !bucket || bucket > 8192 || tokens.minSizes[1] != bucket ||
      header.dims != 1 || header.maxSizes[0] != 1 || header.minSizes[0] != 1)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i) {
    const auto& result = input->outputTensors[i].geometry;
    if (result.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (result.dims != 1 || result.maxSizes[0] != bucket || result.minSizes[0] != bucket)
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = bucket;
  for (unsigned i = 0; i < 2; ++i) {
    output->inputTensorAccessPattern[i].allRequired = 1;
    output->outputTensorAccessPattern[i].allRequired = 1;
  }
  const int32_t params[] = {vocabulary, scheduler_owned ? 1 : 0};
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, params, sizeof(params));
  const unsigned size = static_cast<unsigned>(&_binary_token_indices_i32_o_end - &_binary_token_indices_i32_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_token_indices_i32_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_cast_f32_to_bf16(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& source = input->inputTensors[0].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (source.dataType != DATA_F32 || result.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (source.dims != 2 || !source.maxSizes[0] || source.maxSizes[0] > INT_MAX ||
      !source.maxSizes[1] || source.maxSizes[1] > 8192 ||
      source.minSizes[0] != source.maxSizes[0] || source.minSizes[1] != source.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (result.dims != 2 || result.maxSizes[0] != source.maxSizes[0] ||
      result.maxSizes[1] != source.maxSizes[1] || result.minSizes[0] != result.maxSizes[0] ||
      result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (source.maxSizes[0] + 127) / 128;
  output->indexSpaceGeometry[1] = source.maxSizes[1];
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = 128;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = 127;
    access.mapping[1].indexSpaceDim = 1;
    access.mapping[1].a = 1;
    access.mapping[1].start_b = 0;
    access.mapping[1].end_b = 0;
  }
  output->kernel.paramsNr = 0;
  const unsigned size = static_cast<unsigned>(&_binary_cast_f32_to_bf16_o_end -
                                               &_binary_cast_f32_to_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_cast_f32_to_bf16_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_index_put_i64(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0 &&
      (input->nodeParams.nodeParamsSize != sizeof(int32_t) || !input->nodeParams.nodeParams))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& indices = input->inputTensors[0].geometry;
  const auto& values = input->inputTensors[1].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (indices.dataType != DATA_I32 || values.dataType != DATA_I32 || result.dataType != DATA_I32)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (indices.dims != 1 || !indices.maxSizes[0] || indices.maxSizes[0] > INT_MAX ||
      !indices.minSizes[0] || indices.minSizes[0] > indices.maxSizes[0] ||
      values.dims != 2 || values.maxSizes[0] != 2 || values.minSizes[0] != 2 ||
      values.maxSizes[1] != indices.maxSizes[0] || values.minSizes[1] != indices.minSizes[0])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (result.dims != 2 || result.maxSizes[0] != 2 || result.minSizes[0] != 2 ||
      !result.maxSizes[1] || result.maxSizes[1] > INT_MAX || result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  if (input->nodeParams.nodeParamsSize == sizeof(int32_t)) {
    int32_t slots = 0;
    std::memcpy(&slots, input->nodeParams.nodeParams, sizeof(slots));
    if (slots <= 0 || static_cast<uint64_t>(slots) != result.maxSizes[1])
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = 1;
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->kernel.paramsNr = 0;
  const unsigned size = static_cast<unsigned>(&_binary_index_put_i64_o_end - &_binary_index_put_i64_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_index_put_i64_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_paged_slots(HabanaKernelParams* input, HabanaKernelInstantiation* output,
                                       bool release) {
  const unsigned inputs = 3, outputs = release ? 2 : 1;
  if (input->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t page_size = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(page_size) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&page_size, input->nodeParams.nodeParams, sizeof(page_size));
  if (page_size <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  for (unsigned i = 0; i < inputs + outputs; ++i) {
    const auto& tensor = i < inputs ? input->inputTensors[i].geometry : input->outputTensors[i - inputs].geometry;
    if (tensor.dataType != DATA_I32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    for (unsigned dim = 0; dim < tensor.dims; ++dim)
      if (!tensor.maxSizes[dim] || tensor.minSizes[dim] != tensor.maxSizes[dim])
        return i < inputs ? GLUE_INCOMPATIBLE_INPUT_SIZE : GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  const auto& slots = input->inputTensors[0].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (slots.dims != 2 || slots.maxSizes[0] != 2 || result.dims != 2 || result.maxSizes[0] != 2)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (release) {
    const auto& metadata = input->inputTensors[1].geometry;
    const auto& seen_bitset = input->inputTensors[2].geometry;
    const auto& count = input->outputTensors[1].geometry;
    if (slots.maxSizes[1] > INT_MAX || result.maxSizes[1] != slots.maxSizes[1] ||
        metadata.dims != 1 || metadata.maxSizes[0] != 3 ||
        seen_bitset.dims != 1 || !seen_bitset.maxSizes[0] || seen_bitset.maxSizes[0] > INT_MAX ||
        count.dims != 1 || count.maxSizes[0] != 1)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  } else {
    const auto& pages = input->inputTensors[1].geometry;
    const auto& metadata = input->inputTensors[2].geometry;
    if (slots.maxSizes[1] > INT_MAX || pages.dims != 2 || pages.maxSizes[0] != 2 ||
        pages.maxSizes[1] > INT_MAX || metadata.dims != 2 || metadata.maxSizes[0] != 4 ||
        metadata.maxSizes[1] != slots.maxSizes[1] || result.maxSizes[1] > INT_MAX)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = release ? 1 : slots.maxSizes[1];
  for (unsigned i = 0; i < inputs; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < outputs; ++i) output->outputTensorAccessPattern[i].allRequired = 1;
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &page_size, sizeof(page_size));
  auto* start = release ? &_binary_paged_free_i32_o_start : &_binary_paged_alloc_i32_o_start;
  auto* end = release ? &_binary_paged_free_i32_o_end : &_binary_paged_alloc_i32_o_end;
  const unsigned size = static_cast<unsigned>(end - start), capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, start, size);
  return GLUE_SUCCESS;
}
template <bool Flag, GlueCodeReturn (*Fn)(HabanaKernelParams*, HabanaKernelInstantiation*, bool)>
GlueCodeReturn instantiate_with_flag(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  return Fn(input, output, Flag);
}

template <unsigned ExpectedInputs>
GlueCodeReturn infer_rotary(DeviceId, const ShapeInferenceParams* input, ShapeInferenceOutput* output) {
  if (!output || input->inputTensorsNr != ExpectedInputs || input->outputTensorsNr != 2 ||
      !input->inputTensors || !input->inputTensors[0] || !input->inputTensors[1] ||
      !output->outputTensors || !output->outputTensors[0] || !output->outputTensors[1])
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  output->outputTensors[0]->geometry = input->inputTensors[0]->geometry;
  output->outputTensors[1]->geometry = input->inputTensors[1]->geometry;
  return GLUE_SUCCESS;
}

GlueCodeReturn infer_tile_indices(DeviceId, const ShapeInferenceParams* input,
                                  ShapeInferenceOutput* output) {
  if (!output || input->inputTensorsNr != 4 || input->outputTensorsNr != 1 ||
      !input->inputTensors || !input->inputTensors[2] || !output->outputTensors ||
      !output->outputTensors[0])
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  auto& shape = output->outputTensors[0]->geometry;
  shape.dims = 2;
  shape.maxSizes[0] = shape.minSizes[0] = 64;
  shape.maxSizes[1] = input->inputTensors[2]->geometry.maxSizes[1];
  shape.minSizes[1] = input->inputTensors[2]->geometry.minSizes[1];
  return GLUE_SUCCESS;
}

GlueCodeReturn infer_index_put_i64(DeviceId, const ShapeInferenceParams* input,
                                   ShapeInferenceOutput* output) {
  if (!output || input->inputTensorsNr != 2 || input->outputTensorsNr != 1 ||
      !input->inputTensors || !output->outputTensors || !output->outputTensors[0])
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->nodeParams.nodeParamsSize == 0) return GLUE_SUCCESS;
  if (!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != sizeof(int32_t))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  int32_t slots = 0;
  std::memcpy(&slots, input->nodeParams.nodeParams, sizeof(slots));
  if (slots <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  auto& out_g = output->outputTensors[0]->geometry;
  out_g.dims = 2;
  out_g.maxSizes[0] = out_g.minSizes[0] = 2;
  out_g.maxSizes[1] = out_g.minSizes[1] = static_cast<uint64_t>(slots);
  return GLUE_SUCCESS;
}

GlueCodeReturn infer_rank_sum(DeviceId, const ShapeInferenceParams* input,
                              ShapeInferenceOutput* output) {
  if (!output || input->inputTensorsNr != 1 || input->outputTensorsNr != 1 ||
      !input->inputTensors || !input->inputTensors[0] ||
      !output->outputTensors || !output->outputTensors[0])
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  int32_t world_size = 0;
  if (!input->nodeParams.nodeParams || input->nodeParams.nodeParamsSize != sizeof(world_size))
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&world_size, input->nodeParams.nodeParams, sizeof(world_size));
  if (world_size < 2 || world_size > 8) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& in_g = input->inputTensors[0]->geometry;
  const auto w = static_cast<uint64_t>(world_size);
  // In ShapeInferenceParams, GC uses maxSizes for whichever shape is being inferred
  // (min, max, or actual), while minSizes is unioned with element strides.
  if (in_g.dims != 1 || in_g.maxSizes[0] < w || in_g.maxSizes[0] % w != 0)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  auto& out_g = output->outputTensors[0]->geometry;
  out_g.dims = 1;
  out_g.maxSizes[0] = in_g.maxSizes[0] / w;
  out_g.minSizes[0] = (in_g.minSizes[0] >= w && in_g.minSizes[0] <= in_g.maxSizes[0])
                          ? (in_g.minSizes[0] / w)
                          : 1;
  return GLUE_SUCCESS;
}

struct KernelDescriptor {
  const char* name;
  GlueCodeReturn (*instantiate)(HabanaKernelParams*, HabanaKernelInstantiation*);
  unsigned io_arity;
  GlueCodeReturn (*infer_shape)(DeviceId, const ShapeInferenceParams*, ShapeInferenceOutput*) = nullptr;
  uint64_t name_hash = 0;
  bool dynamic_shapes = false;
};

constexpr KernelDescriptor kKernels[] = {
    {kAdd, &instantiate_add_bf16, 3},
    {kFusedNorm, &instantiate_with_flag<true, &instantiate_norm>, 5},
    {kNorm, &instantiate_with_flag<false, &instantiate_norm>, 3},
    {kSilu, &instantiate_silu, 2},
    {kRotary, &instantiate_rotary, 7, &infer_rotary<5>, 0x003dd119de37e582ULL, true},
    {kPagedMetadata, &instantiate_with_flag<false, &instantiate_paged_metadata>, 10},
    {kTokenIndices, &instantiate_with_flag<false, &instantiate_token_indices>, 4},
    {kCastF32ToBf16, &instantiate_cast_f32_to_bf16, 2},
    {kIndexPutI64, &instantiate_index_put_i64, 3, &infer_index_put_i64, 0x008b5fd7d9960ae4ULL, true},
    {kSchedulerMetadata, &instantiate_with_flag<true, &instantiate_paged_metadata>, 10},
    {kSchedulerTokens, &instantiate_with_flag<true, &instantiate_token_indices>, 4},
    {generic_attention::kSoftmax, &generic_attention::instantiate_softmax, 6},
    {generic_attention::kMerge, &instantiate_with_flag<true, &generic_attention::instantiate_online>, 9},
    {generic_attention::kFinalize, &instantiate_with_flag<false, &generic_attention::instantiate_online>, 3},
    {generic_attention::kHeadStore, &generic_attention::instantiate_head_store, 6},
    {generic_attention::kDecodeExp, &instantiate_with_flag<true, &generic_attention::instantiate_decode>, 5},
    {generic_attention::kDecodeNormalize, &instantiate_with_flag<false, &generic_attention::instantiate_decode>, 3},
    {generic_attention::kDirectSoftmax, &generic_attention::instantiate_direct_softmax, 4},
    {generic_attention::kTileIndices, &generic_attention::instantiate_tile_indices, 5,
     &infer_tile_indices, 0x005d2ca4a663d7b1ULL, true},
    {kPagedAlloc, &instantiate_with_flag<false, &instantiate_paged_slots>, 4},
    {kPagedFree, &instantiate_with_flag<true, &instantiate_paged_slots>, 5},
    {kArgmaxBf16, &instantiate_argmax<ArgmaxKind::part_bf16>, 4},
    {kArgmaxF32, &instantiate_argmax<ArgmaxKind::part_f32>, 4},
    {kArgmaxFinish, &instantiate_argmax<ArgmaxKind::finish_i32>, 4},
    {generic_fp8::kQuant, &generic_fp8::instantiate_quant, 3},
    {generic_fp8::kReduce, &generic_fp8::instantiate_reduce, 5},
    {generic_fp8::kTokenQuant, &generic_fp8::instantiate_token_quant, 3},
    {generic_fp8::kChannelFinish, &generic_fp8::instantiate_channel_finish, 5},
    {generic_fp8::kActRowScale, &generic_fp8::instantiate_act_row_scale, 2},
    {generic_fp8::kWeightBlkScale, &generic_fp8::instantiate_weight_blk_scale, 2},
    {generic_fp8::kQuantAct1k, &generic_fp8::instantiate_quant_act_1k, 3},
    {generic_fp8::kDequantWeight1k, &generic_fp8::instantiate_dequant_weight_1k, 4},
    {generic_fp8::kEpilogue, &instantiate_with_flag<false, &generic_fp8::instantiate_epilogue>, 4},
    {generic_fp8::kEpilogueBias, &instantiate_with_flag<true, &generic_fp8::instantiate_epilogue>, 5},
    {kRotaryPos, &instantiate_rotary_pos, 6, &infer_rotary<4>, 0x004eb228af48f693ULL, true},
    {kEwiseUnaryBf16, &instantiate_with_flag<true, &instantiate_ewise_unary>, 2},
    {kEwiseUnaryF32, &instantiate_with_flag<false, &instantiate_ewise_unary>, 2},
    {kGeluAndMul, &instantiate_gelu_and_mul, 2},
    {kSampleVocabBf16, &instantiate_with_flag<true, &instantiate_sample_vocab>, 7},
    {kSampleVocabF32, &instantiate_with_flag<false, &instantiate_sample_vocab>, 7},
    {kApplyVocabBitmaskBf16, &instantiate_with_flag<true, &instantiate_apply_vocab_bitmask>, 3},
    {kApplyVocabBitmaskF32, &instantiate_with_flag<false, &instantiate_apply_vocab_bitmask>, 3},
    {kRankSumBf16, &instantiate_with_flag<true, &instantiate_rank_sum>, 2,
     &infer_rank_sum, 0x006f3db5b774e8c2ULL, true},
    {kRankSumF32, &instantiate_with_flag<false, &instantiate_rank_sum>, 2,
     &infer_rank_sum, 0x007a4ec6c885f9d3ULL, true},
};
static_assert(std::size(kKernels) == 44);

const KernelDescriptor* find_kernel(const char* name) {
  if (!name) return nullptr;
  for (const auto& entry : kKernels) {
    if (std::strcmp(name, entry.name) == 0) return &entry;
  }
  return nullptr;
}
}  // namespace

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  const uint32_t capacity = *count;
  constexpr uint32_t total = static_cast<uint32_t>(std::size(kKernels));
  *count = device == DEVICE_ID_GAUDI2 ? total : 0;
  if (!guids || !*count) return GLUE_SUCCESS;
  const uint32_t emit = std::min(capacity, total);
  for (uint32_t i = 0; i < emit; ++i) {
    std::strcpy(guids[i].name, kKernels[i].name);
    if (kKernels[i].dynamic_shapes) {
      guids[i].nameHash.Value = kKernels[i].name_hash;
      guids[i].kernelProperties = 0;
      guids[i].supportsDynamicShapes = 1;
    }
  }
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  const auto* kernel = find_kernel(input->guid.name);
  if (!kernel) return GLUE_NODE_NOT_FOUND;
  return kernel->instantiate(input, output);
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* input,
                                              ShapeInferenceOutput* output) {
  const auto* kernel = find_kernel(input->pGuid ? input->pGuid->name : input->guid.name);
  if (!kernel) return GLUE_NODE_NOT_FOUND;
  return kernel->infer_shape ? kernel->infer_shape(device, input, output) : GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* input,
                                                   NodeDataLayouts* layouts, uint32_t* count) {
  const auto* kernel = find_kernel(input->guid.name);
  if (!kernel) return GLUE_NODE_NOT_FOUND;
  if (input->inputTensorNr + input->outputTensorNr != kernel->io_arity)
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const auto capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < input->inputTensorNr; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < input->outputTensorNr; ++i)
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
