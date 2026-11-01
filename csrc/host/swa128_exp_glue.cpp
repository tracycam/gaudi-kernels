// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_swa_o_start, _binary_swa_o_end;
static constexpr char guid[] = "gk_swa128_avgroup_group_v0";

static bool shape(const Tensor& t, TensorDataType type, std::initializer_list<unsigned long> dims) {
  if (t.geometry.dataType != type || t.geometry.dims != dims.size()) return false;
  unsigned i = 0;
  for (auto size : dims) {
    if (t.geometry.maxSizes[i] != size || t.geometry.minSizes[i] != size) return false;
    ++i;
  }
  return true;
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* out) {
  *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
  if (out && *count) std::strcpy(out[0].name, guid);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
  if (std::strcmp(in->guid.name, guid)) return GLUE_NODE_NOT_FOUND;
  if (in->inputTensorNr != 7) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (in->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (in->nodeParams.nodeParamsSize != 4 || !in->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float scale = *static_cast<float*>(in->nodeParams.nodeParams);
  if (!std::isfinite(scale) || scale <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  bool flat = in->inputTensors[0].geometry.maxSizes[0] != 192;
  unsigned long heads = flat ? in->inputTensors[0].geometry.maxSizes[0] / 192
                             : in->inputTensors[0].geometry.maxSizes[1];
  unsigned long slots = in->inputTensors[1].geometry.maxSizes[2];
  if (!heads || heads > 16 || !slots || slots % 128 || slots > INT_MAX / 192)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (!(flat ? shape(in->inputTensors[0], DATA_BF16, {heads * 192, 1, 1})
              : shape(in->inputTensors[0], DATA_BF16, {192, heads, 1})) ||
      !shape(in->inputTensors[1], DATA_BF16, {192, 1, slots}) ||
      !shape(in->inputTensors[2], DATA_BF16, {128, 1, slots}) ||
      !shape(in->inputTensors[6], DATA_BF16, {heads})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  unsigned long pages = in->inputTensors[3].geometry.maxSizes[0];
  if (!pages || pages > INT_MAX || !shape(in->inputTensors[3], DATA_I32, {pages}) ||
      !shape(in->inputTensors[4], DATA_I32, {pages}) ||
      !(shape(in->inputTensors[5], DATA_I32, {1}) || shape(in->inputTensors[5], DATA_I32, {1, 1})))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (!shape(in->outputTensors[0], DATA_BF16, {128, heads, 1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  out->indexSpaceRank = 1;
  out->indexSpaceGeometry[0] = heads;
  for (unsigned i = 0; i < 7; ++i) out->inputTensorAccessPattern[i].allRequired = 1;
  auto& p = out->outputTensorAccessPattern[0];
  p.allRequired = 0;
  p.mapping[0] = {0, 0, 0, 127}; p.mapping[1] = {0, 1, 0, 0}; p.mapping[2] = {0, 0, 0, 0};
  out->kernel.paramsNr = 1;
  std::memcpy(out->kernel.scalarParams, &scale, 4);
  unsigned capacity = out->kernel.elfSize;
  out->kernel.elfSize = &_binary_swa_o_end - &_binary_swa_o_start;
  if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf, &_binary_swa_o_start, out->kernel.elfSize);
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* in, ShapeInferenceOutput*) {
  return std::strcmp(in->pGuid ? in->pGuid->name : in->guid.name, guid) ? GLUE_NODE_NOT_FOUND : GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in, NodeDataLayouts* layouts, uint32_t* count) {
  if (std::strcmp(in->guid.name, guid)) return GLUE_NODE_NOT_FOUND;
  auto capacity = *count; *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < in->inputTensorNr; ++i) std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < in->outputTensorNr; ++i) std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
