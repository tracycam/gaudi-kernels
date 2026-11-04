// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <climits>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_add_bf16_o_start, _binary_add_bf16_o_end;

namespace {
constexpr const char* kAdd = "gk_add_bf16_v1";
bool named(const char* name) { return name && std::strcmp(name, kAdd) == 0; }
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
  if (guids && capacity && *count) std::strcpy(guids[0].name, kAdd);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  if (!named(input->guid.name)) return GLUE_NODE_NOT_FOUND;
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
  return named(input->pGuid ? input->pGuid->name : input->guid.name)
             ? GLUE_SUCCESS : GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* input,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (!named(input->guid.name)) return GLUE_NODE_NOT_FOUND;
  const auto capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < 2; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  std::memset(layouts[0].outputs[0].layout, 'x', sizeof(layouts[0].outputs[0].layout));
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
