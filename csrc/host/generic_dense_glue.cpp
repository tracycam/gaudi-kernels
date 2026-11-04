// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <climits>
#include <cmath>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_add_bf16_o_start, _binary_add_bf16_o_end;
extern "C" unsigned char _binary_fused_add_rms_norm_bf16_o_start, _binary_fused_add_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_rms_norm_bf16_o_start, _binary_rms_norm_bf16_o_end;
extern "C" unsigned char _binary_silu_and_mul_bf16_o_start, _binary_silu_and_mul_bf16_o_end;

namespace {
constexpr const char* kAdd = "gk_add_bf16_v1";
constexpr const char* kFusedNorm = "gk_fused_add_rms_norm_bf16_v1";
constexpr const char* kNorm = "gk_rms_norm_bf16_v1";
constexpr const char* kSilu = "gk_silu_and_mul_bf16_v1";
int kind(const char* name) {
  if (!name) return -1;
  if (std::strcmp(name, kAdd) == 0) return 0;
  if (std::strcmp(name, kFusedNorm) == 0) return 1;
  if (std::strcmp(name, kNorm) == 0) return 2;
  if (std::strcmp(name, kSilu) == 0) return 3;
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
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 4 : 0;
  if (guids && *count && capacity > 0) std::strcpy(guids[0].name, kAdd);
  if (guids && *count && capacity > 1) std::strcpy(guids[1].name, kFusedNorm);
  if (guids && *count && capacity > 2) std::strcpy(guids[2].name, kNorm);
  if (guids && *count && capacity > 3) std::strcpy(guids[3].name, kSilu);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  const int id = kind(input->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  if (id == 1 || id == 2) return instantiate_norm(input, output, id == 1);
  if (id == 3) return instantiate_silu(input, output);
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
