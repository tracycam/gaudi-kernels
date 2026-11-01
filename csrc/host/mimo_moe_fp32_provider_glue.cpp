// SPDX-License-Identifier: Apache-2.0
// The M=1 MiMo graph uses only these two committed expert kernels.
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstring>

using namespace tpc_lib_api;

extern "C" unsigned char _binary_mimo_compact_gate_o_start;
extern "C" unsigned char _binary_mimo_compact_gate_o_end;
extern "C" unsigned char _binary_mimo_combine_f32_o_start;
extern "C" unsigned char _binary_mimo_combine_f32_o_end;

namespace {
constexpr const char* kNames[] = {"gk_mimo_moe_gate_fp32_v1", "pf_combine_f32"};

int kind(const char* name) {
  if (!name) return -1;
  for (int i = 0; i < 2; ++i) if (!std::strcmp(name, kNames[i])) return i;
  return -1;
}

bool shape(const TensorGeometry& value, TensorDataType type, uint64_t x, uint64_t y) {
  return value.dataType == type && value.dims == 2 &&
         value.minSizes[0] == x && value.maxSizes[0] == x &&
         value.minSizes[1] == y && value.maxSizes[1] == y;
}
}  // namespace

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
  if (!count) return GLUE_FAILED;
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 2 : 0;
  if (guids) for (uint32_t i = 0; i < capacity && i < *count; ++i)
    std::strcpy(guids[i].name, kNames[i]);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  if (!input || !output) return GLUE_FAILED;
  const int id = kind(input->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  const unsigned expected_inputs = id == 0 ? 2 : 3;
  if (input->inputTensorNr != expected_inputs || !input->inputTensors)
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1 || !input->outputTensors)
    return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != sizeof(int32_t) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  int32_t parameter = 0;
  std::memcpy(&parameter, input->nodeParams.nodeParams, sizeof(parameter));
  if (parameter != (id == 0 ? 3 : 8)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;

  if (id == 0) {
    if (!shape(input->inputTensors[0].geometry, DATA_F32, 12288, 1) ||
        !shape(input->inputTensors[1].geometry, DATA_I32, 8, 1))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(input->outputTensors[0].geometry, DATA_BF16, 256, 8))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  } else {
    if (!shape(input->inputTensors[0].geometry, DATA_F32, 49152, 1) ||
        !shape(input->inputTensors[1].geometry, DATA_F32, 8, 1) ||
        !shape(input->inputTensors[2].geometry, DATA_U8, 256, 2))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(input->outputTensors[0].geometry, DATA_F32, 6144, 1))
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
  if (!output->inputTensorAccessPattern || !output->outputTensorAccessPattern)
    return GLUE_FAILED;

  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = id == 0 ? 8 : 48;
  for (unsigned i = 0; i < expected_inputs; ++i)
    output->inputTensorAccessPattern[i].allRequired = 1;
  auto& result = output->outputTensorAccessPattern[0];
  if (id == 0) {
    auto& partial = output->inputTensorAccessPattern[0];
    partial.allRequired = 0;
    partial.mapping[0].indexSpaceDim = 0;
    partial.mapping[0].a = 1536;
    partial.mapping[0].start_b = 0;
    partial.mapping[0].end_b = 1535;
    partial.mapping[1].indexSpaceDim = 0;
    partial.mapping[1].a = 0;
    partial.mapping[1].start_b = partial.mapping[1].end_b = 0;
    result.allRequired = 0;
    result.mapping[0].indexSpaceDim = 0;
    result.mapping[0].a = 0;
    result.mapping[0].start_b = 0;
    result.mapping[0].end_b = 255;
    result.mapping[1].indexSpaceDim = 0;
    result.mapping[1].a = 1;
    result.mapping[1].start_b = result.mapping[1].end_b = 0;
  } else {
    result.allRequired = 1;
  }
  output->kernel.paramsNr = 1;
  output->kernel.scalarParams[0] = static_cast<uint32_t>(parameter);
  auto* begin = id == 0 ? &_binary_mimo_compact_gate_o_start : &_binary_mimo_combine_f32_o_start;
  auto* end = id == 0 ? &_binary_mimo_compact_gate_o_end : &_binary_mimo_combine_f32_o_end;
  const unsigned capacity = output->kernel.elfSize;
  const unsigned size = static_cast<unsigned>(end - begin);
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  if (!output->kernel.kernelElf) return GLUE_FAILED;
  std::memcpy(output->kernel.kernelElf, begin, size);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* input,
                                              ShapeInferenceOutput*) {
  if (!input) return GLUE_FAILED;
  return kind(input->pGuid ? input->pGuid->name : input->guid.name) >= 0
             ? GLUE_SUCCESS : GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* input,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (!input || !count) return GLUE_FAILED;
  if (kind(input->guid.name) < 0) return GLUE_NODE_NOT_FOUND;
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
