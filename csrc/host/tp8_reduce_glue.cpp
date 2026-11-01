// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_sum_bf16_bf16_o_start, _binary_sum_bf16_bf16_o_end;
extern "C" unsigned char _binary_sum_f32_bf16_o_start, _binary_sum_f32_bf16_o_end;
static const char* names[] = {"pf_sum_bf16_bf16", "pf_sum_f32_bf16"};

static int kind(const char* name) {
  for (int i = 0; i < 2; ++i) if (!std::strcmp(name, names[i])) return i;
  return -1;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  const unsigned capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 2 : 0;
  if (info) for (unsigned i = 0; i < capacity && i < *count; ++i) std::strcpy(info[i].name, names[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                HabanaKernelInstantiation* out) {
  const int variant = kind(in->guid.name);
  if (variant < 0) return GLUE_NODE_NOT_FOUND;
  if (in->inputTensorNr != 1 || in->outputTensorNr != 1 ||
      in->nodeParams.nodeParamsSize != sizeof(int32_t) || !in->nodeParams.nodeParams)
    return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const auto& input = in->inputTensors[0].geometry;
  const auto& output = in->outputTensors[0].geometry;
  int32_t ranks;
  std::memcpy(&ranks, in->nodeParams.nodeParams, sizeof(ranks));
  if (ranks != 8 || input.dims != 2 || output.dims != 2 ||
      input.dataType != (variant == 0 ? DATA_BF16 : DATA_F32) ||
      output.dataType != DATA_BF16 || output.maxSizes[0] % 128 ||
      input.maxSizes[0] != output.maxSizes[0] ||
      input.maxSizes[1] != output.maxSizes[1] * 8)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->inputTensorAccessPattern[0].allRequired = 1;
  out->outputTensorAccessPattern[0].allRequired = 1;
  out->indexSpaceRank = 1;
  out->indexSpaceGeometry[0] = output.maxSizes[0] * output.maxSizes[1] / 128;
  out->kernel.paramsNr = 1;
  std::memcpy(out->kernel.scalarParams, &ranks, sizeof(ranks));
  unsigned char* starts[] = {&_binary_sum_bf16_bf16_o_start, &_binary_sum_f32_bf16_o_start};
  unsigned char* ends[] = {&_binary_sum_bf16_bf16_o_end, &_binary_sum_f32_bf16_o_end};
  const unsigned capacity = out->kernel.elfSize;
  out->kernel.elfSize = ends[variant] - starts[variant];
  if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf, starts[variant], out->kernel.elfSize);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* in,
                                             ShapeInferenceOutput*) {
  return kind(in->pGuid ? in->pGuid->name : in->guid.name) < 0 ? GLUE_NODE_NOT_FOUND : GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (kind(in->guid.name) < 0) return GLUE_NODE_NOT_FOUND;
  const unsigned capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < in->inputTensorNr; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < in->outputTensorNr; ++i)
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion() { return 1; }
