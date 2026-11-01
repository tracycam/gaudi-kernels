// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_block_o_start, _binary_block_o_end;
extern "C" unsigned char _binary_reduce_o_start, _binary_reduce_o_end;
static const char* guids[] = {"gk_full_diffkv_block_fp32_v0", "gk_full_diffkv_reduce_fp32_v0"};
static int kind(const char* name) {
  for (int i = 0; i < 2; ++i) if (!std::strcmp(name, guids[i])) return i;
  return -1;
}
static bool shape(const Tensor& tensor, TensorDataType type,
                  std::initializer_list<unsigned long> dims) {
  if (tensor.geometry.dataType != type || tensor.geometry.dims != dims.size()) return false;
  unsigned i = 0;
  for (auto value : dims) {
    if (tensor.geometry.maxSizes[i] != value || tensor.geometry.minSizes[i] != value) return false;
    ++i;
  }
  return true;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  *count = device == DEVICE_ID_GAUDI2 ? 2 : 0;
  if (info) for (unsigned i = 0; i < *count; ++i) std::strcpy(info[i].name, guids[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                HabanaKernelInstantiation* out) {
  const int variant = kind(in->guid.name);
  if (variant < 0) return GLUE_NODE_NOT_FOUND;
  const unsigned inputs = variant == 0 ? 6 : 3;
  const unsigned outputs = variant == 0 ? 3 : 1;
  if (in->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (in->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  constexpr unsigned heads = 16;
  const unsigned long pages = in->inputTensors[variant == 0 ? 3 : 0].geometry.maxSizes[0] /
                              (variant == 0 ? 1 : heads);
  if (!pages || pages > 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (variant == 0) {
    const unsigned long slots = in->inputTensors[1].geometry.maxSizes[2];
    if (!slots || slots % 128 || slots > INT_MAX / 192 ||
        !shape(in->inputTensors[1], DATA_BF16, {192, 1, slots}) ||
        !shape(in->inputTensors[0], DATA_BF16, {192, heads, 1}) ||
        !shape(in->inputTensors[2], DATA_BF16, {128, 1, slots}) ||
        !shape(in->inputTensors[3], DATA_I32, {pages}) ||
        !shape(in->inputTensors[4], DATA_I32, {pages}) ||
        !shape(in->inputTensors[5], DATA_I32, {pages}) ||
        !shape(in->outputTensors[0], DATA_F32, {pages * heads}) ||
        !shape(in->outputTensors[1], DATA_F32, {pages * heads}) ||
        !shape(in->outputTensors[2], DATA_F32, {128, pages, heads})) {
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = pages;
    out->indexSpaceGeometry[1] = heads;
  } else {
    if (!shape(in->inputTensors[0], DATA_F32, {pages * heads}) ||
        !shape(in->inputTensors[1], DATA_F32, {pages * heads}) ||
        !shape(in->inputTensors[2], DATA_F32, {128, pages, heads}) ||
        !shape(in->outputTensors[0], DATA_BF16, {128, heads, 1})) {
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = heads;
  }
  if (variant == 0) {
    if (in->nodeParams.nodeParamsSize != 4 || !in->nodeParams.nodeParams) {
      return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const float scale = *static_cast<float*>(in->nodeParams.nodeParams);
    if (!std::isfinite(scale) || scale <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &scale, 4);
  } else {
    out->kernel.paramsNr = 0;
  }
  for (unsigned i = 0; i < inputs; ++i) out->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < outputs; ++i) out->outputTensorAccessPattern[i].allRequired = 1;
  unsigned char* starts[] = {&_binary_block_o_start, &_binary_reduce_o_start};
  unsigned char* ends[] = {&_binary_block_o_end, &_binary_reduce_o_end};
  const unsigned capacity = out->kernel.elfSize;
  out->kernel.elfSize = ends[variant] - starts[variant];
  if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf, starts[variant], out->kernel.elfSize);
  return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion() { return 1; }
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
  for (unsigned i = 0; i < in->inputTensorNr; ++i) {
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  }
  for (unsigned i = 0; i < in->outputTensorNr; ++i) {
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  }
  return GLUE_SUCCESS;
}
