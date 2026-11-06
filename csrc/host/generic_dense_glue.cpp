// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_add_bf16_o_start, _binary_add_bf16_o_end;
extern "C" unsigned char _binary_fused_add_rms_norm_bf16_o_start, _binary_fused_add_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_rms_norm_bf16_o_start, _binary_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_silu_and_mul_bf16_o_start, _binary_silu_and_mul_bf16_o_end;
extern "C" unsigned char _binary_kv_store_bf16_o_start, _binary_kv_store_bf16_o_end;
extern "C" unsigned char _binary_paged_gqa_bf16_o_start, _binary_paged_gqa_bf16_o_end;
extern "C" unsigned char _binary_paged_gqa_mme_gather_bf16_o_start, _binary_paged_gqa_mme_gather_bf16_o_end;
extern "C" unsigned char _binary_paged_gqa_mme_softmax_bf16_o_start, _binary_paged_gqa_mme_softmax_bf16_o_end;
extern "C" unsigned char _binary_rotary_bf16_o_start, _binary_rotary_bf16_o_end;
extern "C" unsigned char _binary_paged_metadata_i32_o_start, _binary_paged_metadata_i32_o_end;
extern "C" unsigned char _binary_token_indices_i32_o_start, _binary_token_indices_i32_o_end;
extern "C" unsigned char _binary_cast_f32_to_bf16_o_start, _binary_cast_f32_to_bf16_o_end;
extern "C" unsigned char _binary_index_put_i64_o_start, _binary_index_put_i64_o_end;

namespace {
constexpr const char* kAdd = "gk_add_bf16_v1";
constexpr const char* kFusedNorm = "gk_fused_add_rms_norm_bf16_v1";
constexpr const char* kNorm = "gk_rms_norm_bf16_v1";
constexpr const char* kSilu = "gk_silu_and_mul_bf16_v1";
constexpr const char* kKvStore = "gk_kv_store_bf16_v1";
constexpr const char* kPagedGqa = "gk_paged_gqa_bf16_v1";
constexpr const char* kPagedGqaMmeGather = "gk_paged_gqa_mme_gather_bf16_v1";
constexpr const char* kPagedGqaMmeSoftmax = "gk_paged_gqa_mme_softmax_bf16_v1";
constexpr const char* kRotary = "gk_rotary_bf16_v1";
constexpr const char* kPagedMetadata = "gk_paged_metadata_i32_v1";
constexpr const char* kTokenIndices = "gk_token_indices_i32_v1";
constexpr const char* kSchedulerMetadata = "gk_scheduler_metadata_i32_v1";
constexpr const char* kSchedulerTokens = "gk_scheduler_tokens_i32_v1";
constexpr const char* kCastF32ToBf16 = "gk_cast_f32_to_bf16_v1";
constexpr const char* kIndexPutI64 = "gk_index_put_i64_v1";
int kind(const char* name) {
  if (!name) return -1;
  if (std::strcmp(name, kAdd) == 0) return 0;
  if (std::strcmp(name, kFusedNorm) == 0) return 1;
  if (std::strcmp(name, kNorm) == 0) return 2;
  if (std::strcmp(name, kSilu) == 0) return 3;
  if (std::strcmp(name, kKvStore) == 0) return 4;
  if (std::strcmp(name, kPagedGqa) == 0) return 5;
  if (std::strcmp(name, kRotary) == 0) return 6;
  if (std::strcmp(name, kPagedMetadata) == 0) return 7;
  if (std::strcmp(name, kTokenIndices) == 0) return 8;
  if (std::strcmp(name, kCastF32ToBf16) == 0) return 9;
  if (std::strcmp(name, kIndexPutI64) == 0) return 10;
  if (std::strcmp(name, kPagedGqaMmeGather) == 0) return 11;
  if (std::strcmp(name, kPagedGqaMmeSoftmax) == 0) return 12;
  if (std::strcmp(name, kSchedulerMetadata) == 0) return 14;
  if (std::strcmp(name, kSchedulerTokens) == 0) return 15;
  return -1;
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
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = shape.maxSizes[1];
  for (unsigned i = 0; i < inputs + outputs; ++i) {
    auto& access = i < inputs ? output->inputTensorAccessPattern[i] : output->outputTensorAccessPattern[i - inputs];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
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
  if (key.dims != 2 || !key.maxSizes[0] || key.maxSizes[0] > INT_MAX ||
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
  output->indexSpaceGeometry[0] = (key.maxSizes[0] + 127) / 128;
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

GlueCodeReturn instantiate_paged_gqa(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 7) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params { int32_t head_dim; float scale; } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if ((params.head_dim != 64 && params.head_dim != 128) || !std::isfinite(params.scale) || params.scale <= 0)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& query = input->inputTensors[0].geometry;
  const auto& key = input->inputTensors[1].geometry;
  const auto& value = input->inputTensors[2].geometry;
  const auto& requests = input->inputTensors[3].geometry;
  const auto& positions = input->inputTensors[4].geometry;
  const auto& mapping = input->inputTensors[5].geometry;
  const auto& metadata = input->inputTensors[6].geometry;
  const auto& result = input->outputTensors[0].geometry;
  if (query.dataType != DATA_BF16 || key.dataType != DATA_BF16 || value.dataType != DATA_BF16 ||
      result.dataType != DATA_BF16 || requests.dataType != DATA_I32 || positions.dataType != DATA_I32 ||
      mapping.dataType != DATA_I32 || metadata.dataType != DATA_I32)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  if (query.dims != 2 || !query.maxSizes[0] || query.maxSizes[0] > INT_MAX ||
      query.maxSizes[0] % params.head_dim || !query.maxSizes[1] || query.maxSizes[1] > 4096 ||
      query.minSizes[0] != query.maxSizes[0] || query.minSizes[1] != query.maxSizes[1] ||
      key.dims != 2 || !key.maxSizes[0] || key.maxSizes[0] > INT_MAX ||
      key.maxSizes[0] % params.head_dim || query.maxSizes[0] % key.maxSizes[0] ||
      !key.maxSizes[1] || key.maxSizes[1] > INT_MAX || key.minSizes[0] != key.maxSizes[0] ||
      key.minSizes[1] != key.maxSizes[1] || value.dims != 2 ||
      value.maxSizes[0] != key.maxSizes[0] || value.maxSizes[1] != key.maxSizes[1] ||
      value.minSizes[0] != value.maxSizes[0] || value.minSizes[1] != value.maxSizes[1])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (const auto* vector : {&requests, &positions})
    if (vector->dims != 1 || vector->maxSizes[0] != query.maxSizes[1] || vector->minSizes[0] != vector->maxSizes[0])
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (mapping.dims != 2 || !mapping.maxSizes[0] || mapping.maxSizes[0] > INT_MAX ||
      !mapping.maxSizes[1] || mapping.maxSizes[1] > INT_MAX || mapping.minSizes[0] != mapping.maxSizes[0] ||
      mapping.minSizes[1] != mapping.maxSizes[1] || metadata.dims != 1 || metadata.maxSizes[0] != 1 ||
      metadata.minSizes[0] != 1)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (result.dims != 2 || result.maxSizes[0] != query.maxSizes[0] || result.maxSizes[1] != query.maxSizes[1] ||
      result.minSizes[0] != result.maxSizes[0] || result.minSizes[1] != result.maxSizes[1])
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = query.maxSizes[0] / params.head_dim;
  output->indexSpaceGeometry[1] = query.maxSizes[1];
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = i ? output->outputTensorAccessPattern[0] : output->inputTensorAccessPattern[0];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = params.head_dim;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = params.head_dim - 1;
    access.mapping[1].indexSpaceDim = 1;
    access.mapping[1].a = 1;
    access.mapping[1].start_b = 0;
    access.mapping[1].end_b = 0;
  }
  for (unsigned i = 1; i < 7; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  const unsigned size = static_cast<unsigned>(&_binary_paged_gqa_bf16_o_end - &_binary_paged_gqa_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_paged_gqa_bf16_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_paged_gqa_mme_gather(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 6) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t head_dim = 0;
  if (input->nodeParams.nodeParamsSize != sizeof(head_dim) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&head_dim, input->nodeParams.nodeParams, sizeof(head_dim));
  if (head_dim != 64 && head_dim != 128) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto matches = [](const auto& geometry, auto dtype, std::initializer_list<uint64_t> dims) {
    if (geometry.dataType != dtype || geometry.dims != dims.size()) return false;
    size_t i = 0;
    for (const auto size : dims) {
      if (!size || size > INT_MAX || geometry.maxSizes[i] != size || geometry.minSizes[i] != size) return false;
      ++i;
    }
    return true;
  };
  const auto& key = input->inputTensors[0].geometry;
  const auto& mapping = input->inputTensors[4].geometry;
  const auto& row_requests = input->inputTensors[2].geometry;
  if (key.dims != 2 || mapping.dims != 2 || row_requests.dims != 1 ||
      !key.maxSizes[0] || key.maxSizes[0] % head_dim || row_requests.maxSizes[0] > 4096)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const uint64_t kv_width = key.maxSizes[0], slots = key.maxSizes[1], bucket = row_requests.maxSizes[0];
  const uint64_t context = mapping.maxSizes[0], requests = mapping.maxSizes[1], kv_heads = kv_width / head_dim;
  if (!matches(key, DATA_BF16, {kv_width, slots}) ||
      !matches(input->inputTensors[1].geometry, DATA_BF16, {kv_width, slots}) ||
      !matches(row_requests, DATA_I32, {bucket}) || !matches(input->inputTensors[3].geometry, DATA_I32, {bucket}) ||
      !matches(mapping, DATA_I32, {context, requests}) || !matches(input->inputTensors[5].geometry, DATA_I32, {1}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i)
    if (!matches(input->outputTensors[i].geometry, DATA_BF16,
                 {static_cast<uint64_t>(head_dim), context, kv_heads, bucket}))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = context;
  output->indexSpaceGeometry[1] = kv_heads;
  output->indexSpaceGeometry[2] = bucket;
  for (unsigned i = 0; i < 6; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = 0;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = head_dim - 1;
    for (unsigned dim = 1; dim < 4; ++dim) {
      access.mapping[dim].indexSpaceDim = dim - 1;
      access.mapping[dim].a = 1;
      access.mapping[dim].start_b = 0;
      access.mapping[dim].end_b = 0;
    }
  }
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &head_dim, sizeof(head_dim));
  const unsigned size = static_cast<unsigned>(&_binary_paged_gqa_mme_gather_bf16_o_end -
                                               &_binary_paged_gqa_mme_gather_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_paged_gqa_mme_gather_bf16_o_start, size);
  return GLUE_SUCCESS;
}

GlueCodeReturn instantiate_paged_gqa_mme_softmax(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  struct Params { int32_t kv_heads; float scale; } params{};
  if (input->nodeParams.nodeParamsSize != sizeof(params) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&params, input->nodeParams.nodeParams, sizeof(params));
  if (params.kv_heads <= 0 || !std::isfinite(params.scale) || params.scale <= 0)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto matches = [](const auto& geometry, auto dtype, std::initializer_list<uint64_t> dims) {
    if (geometry.dataType != dtype || geometry.dims != dims.size()) return false;
    size_t i = 0;
    for (const auto size : dims) {
      if (!size || size > INT_MAX || geometry.maxSizes[i] != size || geometry.minSizes[i] != size) return false;
      ++i;
    }
    return true;
  };
  const auto& scores = input->inputTensors[0].geometry;
  if (scores.dims != 3 || scores.maxSizes[2] % params.kv_heads) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const uint64_t context = scores.maxSizes[0], ratio = scores.maxSizes[1], groups = scores.maxSizes[2];
  const uint64_t bucket = groups / params.kv_heads;
  if (bucket > 4096 || !matches(scores, DATA_F32, {context, ratio, groups}) ||
      !matches(input->inputTensors[1].geometry, DATA_I32, {bucket}) ||
      !matches(input->inputTensors[2].geometry, DATA_I32, {1}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = 0; i < 2; ++i)
    if (!matches(input->outputTensors[i].geometry, DATA_BF16, {context, ratio, groups}))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = ratio;
  output->indexSpaceGeometry[1] = groups;
  for (unsigned i = 0; i < 3; ++i) output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    access.allRequired = 0;
    access.mapping[0].indexSpaceDim = 0;
    access.mapping[0].a = 0;
    access.mapping[0].start_b = 0;
    access.mapping[0].end_b = context - 1;
    for (unsigned dim = 1; dim < 3; ++dim) {
      access.mapping[dim].indexSpaceDim = dim - 1;
      access.mapping[dim].a = 1;
      access.mapping[dim].start_b = 0;
      access.mapping[dim].end_b = 0;
    }
  }
  output->kernel.paramsNr = 2;
  std::memcpy(output->kernel.scalarParams, &params, sizeof(params));
  const unsigned size = static_cast<unsigned>(&_binary_paged_gqa_mme_softmax_bf16_o_end -
                                               &_binary_paged_gqa_mme_softmax_bf16_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_paged_gqa_mme_softmax_bf16_o_start, size);
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
      cache.minSizes[1] != cache.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
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
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 15 : 0;
  if (guids && *count && capacity > 0) std::strcpy(guids[0].name, kAdd);
  if (guids && *count && capacity > 1) std::strcpy(guids[1].name, kFusedNorm);
  if (guids && *count && capacity > 2) std::strcpy(guids[2].name, kNorm);
  if (guids && *count && capacity > 3) std::strcpy(guids[3].name, kSilu);
  if (guids && *count && capacity > 4) std::strcpy(guids[4].name, kKvStore);
  if (guids && *count && capacity > 5) std::strcpy(guids[5].name, kPagedGqa);
  if (guids && *count && capacity > 6) std::strcpy(guids[6].name, kRotary);
  if (guids && *count && capacity > 7) std::strcpy(guids[7].name, kPagedMetadata);
  if (guids && *count && capacity > 8) std::strcpy(guids[8].name, kTokenIndices);
  if (guids && *count && capacity > 9) std::strcpy(guids[9].name, kCastF32ToBf16);
  if (guids && *count && capacity > 10) std::strcpy(guids[10].name, kIndexPutI64);
  if (guids && *count && capacity > 11) std::strcpy(guids[11].name, kPagedGqaMmeGather);
  if (guids && *count && capacity > 12) std::strcpy(guids[12].name, kPagedGqaMmeSoftmax);
  if (guids && *count && capacity > 13) std::strcpy(guids[13].name, kSchedulerMetadata);
  if (guids && *count && capacity > 14) std::strcpy(guids[14].name, kSchedulerTokens);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  const int id = kind(input->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  if (id == 1 || id == 2) return instantiate_norm(input, output, id == 1);
  if (id == 3) return instantiate_silu(input, output);
  if (id == 4) return instantiate_kv_store(input, output);
  if (id == 5) return instantiate_paged_gqa(input, output);
  if (id == 6) return instantiate_rotary(input, output);
  if (id == 7 || id == 14) return instantiate_paged_metadata(input, output, id == 14);
  if (id == 8 || id == 15) return instantiate_token_indices(input, output, id == 15);
  if (id == 9) return instantiate_cast_f32_to_bf16(input, output);
  if (id == 10) return instantiate_index_put_i64(input, output);
  if (id == 11) return instantiate_paged_gqa_mme_gather(input, output);
  if (id == 12) return instantiate_paged_gqa_mme_softmax(input, output);
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
                                              ShapeInferenceOutput*) {
  return kind(input->pGuid ? input->pGuid->name : input->guid.name) >= 0
             ? GLUE_SUCCESS : GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* input,
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
