// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include "generic_attention_glue.hpp"
#include "generic_fp8_glue.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_add_bf16_o_start, _binary_add_bf16_o_end;
extern "C" unsigned char _binary_fused_add_rms_norm_bf16_o_start, _binary_fused_add_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_rms_norm_bf16_o_start, _binary_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_silu_and_mul_bf16_o_start, _binary_silu_and_mul_bf16_o_end;
extern "C" unsigned char _binary_kv_store_bf16_o_start, _binary_kv_store_bf16_o_end;
extern "C" unsigned char _binary_rotary_bf16_o_start, _binary_rotary_bf16_o_end;
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
constexpr const char* kKvStore = "gk_kv_store_bf16_v1";
constexpr const char* kRotary = "gk_rotary_bf16_v1";
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
int kind(const char* name) {
  if (!name) return -1;
  if (std::strcmp(name, kAdd) == 0) return 0;
  if (std::strcmp(name, kFusedNorm) == 0) return 1;
  if (std::strcmp(name, kNorm) == 0) return 2;
  if (std::strcmp(name, kSilu) == 0) return 3;
  if (std::strcmp(name, kKvStore) == 0) return 4;
  if (std::strcmp(name, kRotary) == 0) return 6;
  if (std::strcmp(name, kPagedMetadata) == 0) return 7;
  if (std::strcmp(name, kTokenIndices) == 0) return 8;
  if (std::strcmp(name, kCastF32ToBf16) == 0) return 9;
  if (std::strcmp(name, kIndexPutI64) == 0) return 10;
  if (std::strcmp(name, kPagedAlloc) == 0) return 25;
  if (std::strcmp(name, kPagedFree) == 0) return 26;
  if (std::strcmp(name, kArgmaxBf16) == 0) return 27;
  if (std::strcmp(name, kArgmaxF32) == 0) return 28;
  if (std::strcmp(name, kArgmaxFinish) == 0) return 29;
  if (std::strcmp(name, generic_fp8::kQuant) == 0) return 30;
  if (std::strcmp(name, generic_fp8::kReduce) == 0) return 31;
  if (std::strcmp(name, kSchedulerMetadata) == 0) return 14;
  if (std::strcmp(name, kSchedulerTokens) == 0) return 15;
  if (std::strcmp(name, generic_attention::kSoftmax) == 0) return 16;
  if (std::strcmp(name, generic_attention::kMerge) == 0) return 17;
  if (std::strcmp(name, generic_attention::kFinalize) == 0) return 18;
  if (std::strcmp(name, generic_attention::kHeadStore) == 0) return 19;
  if (std::strcmp(name, generic_attention::kDecodeExp) == 0) return 20;
  if (std::strcmp(name, generic_attention::kDecodeNormalize) == 0) return 21;
  if (std::strcmp(name, generic_attention::kDirectSoftmax) == 0) return 22;
  if (std::strcmp(name, generic_attention::kTileIndices) == 0) return 23;
  if (std::strcmp(name, generic_attention::kPack) == 0) return 24;
  return -1;
}

GlueCodeReturn instantiate_argmax(HabanaKernelParams* input, HabanaKernelInstantiation* output, int id) {
  const bool finish = id == 29;
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
  if (finish) {
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
  }
  const auto width = source.maxSizes[0];
  const auto expected_chunk = ((width + 127) / 128 + parts - 1) / parts;
  if (!width || width > INT_MAX || params[0] != static_cast<int>(width) ||
      params[1] != static_cast<int>(expected_chunk) ||
      !generic_attention::shape(input->inputTensors[0], id == 27 ? DATA_BF16 : DATA_F32, {width, rows}) ||
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
  return id == 27 ? generic_attention::binary(output, &_binary_argmax_parts_bf16_o_start,
                                              &_binary_argmax_parts_bf16_o_end)
                  : generic_attention::binary(output, &_binary_argmax_parts_f32_o_start,
                                              &_binary_argmax_parts_f32_o_end);
}

GlueCodeReturn instantiate_norm(HabanaKernelParams* input, HabanaKernelInstantiation* output, bool fused) {
  const unsigned inputs = fused ? 3 : 2;
  const unsigned outputs = fused ? 2 : 1;
  const unsigned gamma = inputs - 1;
  if (input->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != sizeof(float) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float epsilon;
  std::memcpy(&epsilon, input->nodeParams.nodeParams, sizeof(epsilon));
  if (!std::isfinite(epsilon) || epsilon <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
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
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &epsilon, sizeof(epsilon));
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
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (width + 127) / 128;
  output->indexSpaceGeometry[1] = source.maxSizes[1];
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = 128;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = (i ? 0 : width) + 127;
    access.mapping[1].indexSpaceDim = 1;
    access.mapping[1].a = 1;
    access.mapping[1].start_b = 0;
    access.mapping[1].end_b = 0;
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

GlueCodeReturn instantiate_kv_store(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& key = input->inputTensors[0].geometry;
  const auto& value = input->inputTensors[1].geometry;
  const auto& locations = input->inputTensors[2].geometry;
  const auto& metadata = input->inputTensors[3].geometry;
  if (key.dataType != DATA_BF16 || value.dataType != DATA_BF16 || locations.dataType != DATA_I32 ||
      metadata.dataType != DATA_I32 || input->outputTensors[0].geometry.dataType != DATA_BF16 ||
      input->outputTensors[1].geometry.dataType != DATA_BF16)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (key.dims != 2 || !key.maxSizes[0] || key.maxSizes[0] > INT_MAX || key.maxSizes[0] % 128 ||
      !key.maxSizes[1] || key.maxSizes[1] > 4096 || key.minSizes[0] != key.maxSizes[0] ||
      key.minSizes[1] != key.maxSizes[1] || value.dims != 2 ||
      value.maxSizes[0] != key.maxSizes[0] || value.maxSizes[1] != key.maxSizes[1] ||
      value.minSizes[0] != value.maxSizes[0] || value.minSizes[1] != value.maxSizes[1] ||
      locations.dims != 1 || locations.maxSizes[0] != key.maxSizes[1] ||
      locations.minSizes[0] != locations.maxSizes[0] || metadata.dims != 1 ||
      metadata.maxSizes[0] != 1 || metadata.minSizes[0] != 1)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i) {
    const auto& pool = input->outputTensors[i].geometry;
    if (pool.dims != 2 || pool.maxSizes[0] != key.maxSizes[0] || !pool.maxSizes[1] ||
        pool.maxSizes[1] > INT_MAX || pool.minSizes[0] != pool.maxSizes[0] ||
        pool.minSizes[1] != pool.maxSizes[1] ||
        (i == 1 && pool.maxSizes[1] != input->outputTensors[0].geometry.maxSizes[1]))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = key.maxSizes[0] / 128;
  output->indexSpaceGeometry[1] = key.maxSizes[1];
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = output->inputTensorAccessPattern[i];
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
  output->inputTensorAccessPattern[2].allRequired = 1;
  output->inputTensorAccessPattern[3].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->outputTensorAccessPattern[1].allRequired = 1;
  output->kernel.paramsNr = 0;
  const unsigned size = static_cast<unsigned>(&_binary_kv_store_bf16_o_end - &_binary_kv_store_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_kv_store_bf16_o_start, size);
  return GLUE_SUCCESS;
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
        tensor->maxSizes[0] % params.head_dim || !tensor->maxSizes[1] || tensor->maxSizes[1] > 4096 ||
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
  if (locations.dims != 2 || !bucket || bucket > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
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
      tokens.minSizes[0] != tokens.maxSizes[0] || !bucket || bucket > 4096 || tokens.minSizes[1] != bucket ||
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
      !source.maxSizes[1] || source.maxSizes[1] > 4096 ||
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
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& indices = input->inputTensors[0].geometry;
  const auto& values = input->inputTensors[1].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (indices.dataType != DATA_I32 || values.dataType != DATA_I32 || result.dataType != DATA_I32)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (indices.dims != 1 || !indices.maxSizes[0] || indices.maxSizes[0] > INT_MAX ||
      indices.minSizes[0] != indices.maxSizes[0] || values.dims != 2 || values.maxSizes[0] != 2 ||
      values.minSizes[0] != 2 || values.maxSizes[1] != indices.maxSizes[0] ||
      values.minSizes[1] != values.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (result.dims != 2 || result.maxSizes[0] != 2 || result.minSizes[0] != 2 ||
      !result.maxSizes[1] || result.maxSizes[1] > INT_MAX || result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = indices.maxSizes[0];
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
  const unsigned inputs = release ? 2 : 3, outputs = release ? 2 : 1;
  if (input->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t page_size = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(page_size) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&page_size, input->nodeParams.nodeParams, sizeof(page_size));
  if (page_size != 16 && page_size != 64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
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
    const auto& count = input->outputTensors[1].geometry;
    if (slots.maxSizes[1] > 131072 || result.maxSizes[1] != slots.maxSizes[1] ||
        metadata.dims != 1 || metadata.maxSizes[0] != 2 || count.dims != 1 || count.maxSizes[0] != 1)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  } else {
    const auto& pages = input->inputTensors[1].geometry;
    const auto& metadata = input->inputTensors[2].geometry;
    if (slots.maxSizes[1] > 256 || pages.dims != 2 || pages.maxSizes[0] != 2 ||
        pages.maxSizes[1] != 512 || metadata.dims != 2 || metadata.maxSizes[0] != 4 ||
        metadata.maxSizes[1] != slots.maxSizes[1] || result.maxSizes[1] > 4096)
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
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 28 : 0;
  if (guids && *count && capacity > 0) std::strcpy(guids[0].name, kAdd);
  if (guids && *count && capacity > 1) std::strcpy(guids[1].name, kFusedNorm);
  if (guids && *count && capacity > 2) std::strcpy(guids[2].name, kNorm);
  if (guids && *count && capacity > 3) std::strcpy(guids[3].name, kSilu);
  if (guids && *count && capacity > 4) std::strcpy(guids[4].name, kKvStore);
  if (guids && *count && capacity > 5) {
    std::strcpy(guids[5].name, kRotary);
    guids[5].nameHash.Value = 0x003dd119de37e582ULL;
    guids[5].kernelProperties = 0;
    guids[5].supportsDynamicShapes = 1;
  }
  if (guids && *count && capacity > 6) std::strcpy(guids[6].name, kPagedMetadata);
  if (guids && *count && capacity > 7) std::strcpy(guids[7].name, kTokenIndices);
  if (guids && *count && capacity > 8) std::strcpy(guids[8].name, kCastF32ToBf16);
  if (guids && *count && capacity > 9) std::strcpy(guids[9].name, kIndexPutI64);
  if (guids && *count && capacity > 10) std::strcpy(guids[10].name, kSchedulerMetadata);
  if (guids && *count && capacity > 11) std::strcpy(guids[11].name, kSchedulerTokens);
  if (guids && *count && capacity > 12) std::strcpy(guids[12].name, generic_attention::kSoftmax);
  if (guids && *count && capacity > 13) std::strcpy(guids[13].name, generic_attention::kMerge);
  if (guids && *count && capacity > 14) std::strcpy(guids[14].name, generic_attention::kFinalize);
  if (guids && *count && capacity > 15) std::strcpy(guids[15].name, generic_attention::kHeadStore);
  if (guids && *count && capacity > 16) std::strcpy(guids[16].name, generic_attention::kDecodeExp);
  if (guids && *count && capacity > 17) std::strcpy(guids[17].name, generic_attention::kDecodeNormalize);
  if (guids && *count && capacity > 18) std::strcpy(guids[18].name, generic_attention::kDirectSoftmax);
  if (guids && *count && capacity > 19) {
    std::strcpy(guids[19].name, generic_attention::kTileIndices);
    guids[19].nameHash.Value = 0x005d2ca4a663d7b1ULL;
    guids[19].kernelProperties = 0;
    guids[19].supportsDynamicShapes = 1;
  }
  if (guids && *count && capacity > 20) std::strcpy(guids[20].name, generic_attention::kPack);
  if (guids && *count && capacity > 21) std::strcpy(guids[21].name, kPagedAlloc);
  if (guids && *count && capacity > 22) std::strcpy(guids[22].name, kPagedFree);
  if (guids && *count && capacity > 23) std::strcpy(guids[23].name, kArgmaxBf16);
  if (guids && *count && capacity > 24) std::strcpy(guids[24].name, kArgmaxF32);
  if (guids && *count && capacity > 25) std::strcpy(guids[25].name, kArgmaxFinish);
  if (guids && *count && capacity > 26) std::strcpy(guids[26].name, generic_fp8::kQuant);
  if (guids && *count && capacity > 27) std::strcpy(guids[27].name, generic_fp8::kReduce);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  const int id = kind(input->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  if (id == 1 || id == 2) return instantiate_norm(input, output, id == 1);
  if (id == 3) return instantiate_silu(input, output);
  if (id == 4) return instantiate_kv_store(input, output);
  if (id == 6) return instantiate_rotary(input, output);
  if (id == 7 || id == 14) return instantiate_paged_metadata(input, output, id == 14);
  if (id == 8 || id == 15) return instantiate_token_indices(input, output, id == 15);
  if (id == 9) return instantiate_cast_f32_to_bf16(input, output);
  if (id == 10) return instantiate_index_put_i64(input, output);
  if (id == 16) return generic_attention::instantiate_softmax(input, output);
  if (id == 17 || id == 18) return generic_attention::instantiate_online(input, output, id == 17);
  if (id == 19) return generic_attention::instantiate_head_store(input, output);
  if (id == 20 || id == 21) return generic_attention::instantiate_decode(input, output, id == 20);
  if (id == 22) return generic_attention::instantiate_direct_softmax(input, output);
  if (id == 23) return generic_attention::instantiate_tile_indices(input, output);
  if (id == 24) return generic_attention::instantiate_pack(input, output);
  if (id == 25 || id == 26) return instantiate_paged_slots(input, output, id == 26);
  if (id == 27 || id == 28 || id == 29) return instantiate_argmax(input, output, id);
  if (id == 30) return generic_fp8::instantiate_quant(input, output);
  if (id == 31) return generic_fp8::instantiate_reduce(input, output);
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
  const unsigned size = static_cast<unsigned>(&_binary_add_bf16_o_end - &_binary_add_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_add_bf16_o_start, size);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* input,
                                              ShapeInferenceOutput* output) {
  const int id = kind(input->pGuid ? input->pGuid->name : input->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  if (id == 6) {
    if (!output || input->inputTensorsNr != 5 || input->outputTensorsNr != 2 ||
        !input->inputTensors || !input->inputTensors[0] || !input->inputTensors[1] ||
        !output->outputTensors || !output->outputTensors[0] || !output->outputTensors[1])
      return GLUE_INCOMPATIBLE_INPUT_COUNT;
    output->outputTensors[0]->geometry = input->inputTensors[0]->geometry;
    output->outputTensors[1]->geometry = input->inputTensors[1]->geometry;
  }
  if (id == 23) {
    if (!output || input->inputTensorsNr != 4 || input->outputTensorsNr != 1 ||
        !input->inputTensors || !input->inputTensors[2] || !output->outputTensors ||
        !output->outputTensors[0]) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    auto& shape = output->outputTensors[0]->geometry;
    shape.dims = 2;
    shape.maxSizes[0] = shape.minSizes[0] = 64;
    shape.maxSizes[1] = input->inputTensors[2]->geometry.maxSizes[1];
    shape.minSizes[1] = input->inputTensors[2]->geometry.minSizes[1];
  }
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* input,
                                                   NodeDataLayouts* layouts, uint32_t* count) {
  if (kind(input->guid.name) < 0) return GLUE_NODE_NOT_FOUND;
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
