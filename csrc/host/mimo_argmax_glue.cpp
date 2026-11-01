// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>

using namespace tpc_lib_api;

extern "C" unsigned char _binary_mimo_argmax_parts_o_start;
extern "C" unsigned char _binary_mimo_argmax_parts_o_end;
extern "C" unsigned char _binary_mimo_argmax_finish_o_start;
extern "C" unsigned char _binary_mimo_argmax_finish_o_end;

namespace {
constexpr const char* kParts = "gk_mimo_argmax_parts_bf16_v1";
constexpr const char* kFinish = "gk_mimo_argmax_finish_v1";

bool named(const char* actual, const char* expected) {
  return actual && std::strcmp(actual, expected) == 0;
}

bool shape(const TensorGeometry& geometry, TensorDataType type, uint64_t first,
           uint32_t dims) {
  if (geometry.dataType != type || geometry.dims != dims ||
      geometry.maxSizes[0] != first || (dims == 2 && geometry.maxSizes[1] != 1))
    return false;
  for (uint32_t i = 0; i < dims; ++i)
    if (geometry.minSizes[i] != geometry.maxSizes[i]) return false;
  return true;
}
}  // namespace

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count,
                                           GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 2 : 0;
  if (guids && capacity && *count) std::strcpy(guids[0].name, kParts);
  if (guids && capacity > 1 && *count) std::strcpy(guids[1].name, kFinish);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  const bool parts = named(input->guid.name, kParts);
  const bool finish = named(input->guid.name, kFinish);
  if (!parts && !finish) return GLUE_NODE_NOT_FOUND;
  if (input->inputTensorNr != (parts ? 1U : 4U)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != (parts ? 3U : 1U)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  if (parts) {
    if (!shape(input->inputTensors[0].geometry, DATA_BF16, 152576, 2))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(input->outputTensors[0].geometry, DATA_F32, 24, 1) ||
        !shape(input->outputTensors[1].geometry, DATA_I32, 24, 1) ||
        !shape(input->outputTensors[2].geometry, DATA_I32, 24, 1))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  } else {
    if (!shape(input->inputTensors[0].geometry, DATA_F32, 24, 1) ||
        !shape(input->inputTensors[1].geometry, DATA_I32, 24, 1) ||
        !shape(input->inputTensors[2].geometry, DATA_I32, 24, 1) ||
        !shape(input->inputTensors[3].geometry, DATA_I32, 1, 1))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(input->outputTensors[0].geometry, DATA_I32, 1, 1))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = parts ? 24 : 1;
  for (unsigned i = 0; i < input->inputTensorNr; ++i)
    output->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < input->outputTensorNr; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    access.allRequired = parts ? 0 : 1;
    if (parts) {
      access.mapping[0].indexSpaceDim = 0;
      access.mapping[0].a = 1;
      access.mapping[0].start_b = access.mapping[0].end_b = 0;
    }
  }
  output->kernel.paramsNr = 0;
  const auto* begin = parts ? &_binary_mimo_argmax_parts_o_start : &_binary_mimo_argmax_finish_o_start;
  const auto* end = parts ? &_binary_mimo_argmax_parts_o_end : &_binary_mimo_argmax_finish_o_end;
  const unsigned size = static_cast<unsigned>(end - begin);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, begin, size);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* input,
                                              ShapeInferenceOutput*) {
  const char* name = input->pGuid ? input->pGuid->name : input->guid.name;
  return named(name, kParts) || named(name, kFinish) ? GLUE_SUCCESS : GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* input,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (!named(input->guid.name, kParts) && !named(input->guid.name, kFinish))
    return GLUE_NODE_NOT_FOUND;
  const uint32_t capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < input->inputTensorNr; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < input->outputTensorNr; ++i)
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
