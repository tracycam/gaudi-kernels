// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_swiglu_o_start, _binary_swiglu_o_end;
extern "C" unsigned char _binary_residual_rmsnorm_o_start, _binary_residual_rmsnorm_o_end;
extern "C" unsigned char _binary_linear_o_start, _binary_linear_o_end;
static const char* guids[] = {"gk_dense_swiglu_fp32_v1", "gk_dense_residual_rmsnorm_fp32_v1",
                              "gk_dense_linear_fp32_v1"};
static int kind(const char* name) {
  for (int i = 0; i < 3; ++i) if (!std::strcmp(name, guids[i])) return i;
  return -1;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  *count = device == DEVICE_ID_GAUDI2 ? 3 : 0;
  if (info) for (unsigned i = 0; i < *count; ++i) std::strcpy(info[i].name, guids[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                HabanaKernelInstantiation* out) {
  const int variant = kind(in->guid.name);
  if (variant < 0) return GLUE_NODE_NOT_FOUND;
  const unsigned inputs = variant == 0 ? 1 : (variant == 1 ? 3 : 4);
  const unsigned outputs = variant == 1 ? 2 : 1;
  if (in->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (in->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  out->kernel.paramsNr = 0;
  if (variant == 0) {
    const auto& input = in->inputTensors[0].geometry;
    const auto& output = in->outputTensors[0].geometry;
    if (input.dataType != DATA_F32 || input.dims != 2 || input.maxSizes[0] != 4096 ||
        input.maxSizes[1] != 1 || input.minSizes[0] != 4096 || input.minSizes[1] != 1 ||
        output.dataType != DATA_F32 || output.dims != 2 || output.maxSizes[0] != 2048 ||
        output.maxSizes[1] != 1 || output.minSizes[0] != 2048 || output.minSizes[1] != 1) {
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 16;
  } else if (variant == 1) {
    for (unsigned i = 0; i < 2; ++i) {
      const auto& input = in->inputTensors[i].geometry;
      if (input.dataType != DATA_BF16 || input.dims != 2 || input.maxSizes[0] != 6144 ||
          input.maxSizes[1] != 1 || input.minSizes[0] != 6144 || input.minSizes[1] != 1) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
      }
    }
    if (in->inputTensors[2].geometry.dataType != DATA_BF16 ||
        in->inputTensors[2].geometry.dims != 1 ||
        in->inputTensors[2].geometry.maxSizes[0] != 6144 ||
        in->outputTensors[0].geometry.dataType != DATA_BF16 ||
        in->outputTensors[1].geometry.dataType != DATA_F32) {
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (unsigned i = 0; i < 2; ++i) {
      const auto& output = in->outputTensors[i].geometry;
      if (output.dims != 2 || output.maxSizes[0] != 6144 || output.maxSizes[1] != 1 ||
          output.minSizes[0] != 6144 || output.minSizes[1] != 1) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
      }
    }
    if (in->nodeParams.nodeParamsSize != sizeof(float) || !in->nodeParams.nodeParams) {
      return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const float epsilon = *static_cast<const float*>(in->nodeParams.nodeParams);
    if (!std::isfinite(epsilon) || epsilon <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = 1;
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &epsilon, sizeof(epsilon));
  } else {
    const auto& x = in->inputTensors[0].geometry;
    const auto& weight = in->inputTensors[1].geometry;
    const auto& scales = in->inputTensors[2].geometry;
    const auto& bias = in->inputTensors[3].geometry;
    const auto& output = in->outputTensors[0].geometry;
    const unsigned n = weight.maxSizes[1], groups = weight.maxSizes[2], k = groups * 128;
    if (x.dataType != DATA_F32 || x.dims != 2 || x.maxSizes[0] != k || x.maxSizes[1] != 1 ||
        weight.dataType != DATA_F8_143 || weight.dims != 3 || weight.maxSizes[0] != 128 ||
        !groups || !n || n % 128 || (n != 4096 && n != 6144) ||
        scales.dataType != DATA_F32 || scales.dims != 2 || scales.maxSizes[0] != groups ||
        scales.maxSizes[1] != n / 128 || bias.dataType != DATA_F32 || bias.dims != 1 ||
        bias.maxSizes[0] != n || output.dataType != DATA_F32 || output.dims != 2 ||
        output.maxSizes[0] != n || output.maxSizes[1] != 1) {
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = n / 128;
  }
  for (unsigned i = 0; i < inputs; ++i) out->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < outputs; ++i) out->outputTensorAccessPattern[i].allRequired = 1;
  unsigned char* starts[] = {&_binary_swiglu_o_start, &_binary_residual_rmsnorm_o_start,
                             &_binary_linear_o_start};
  unsigned char* ends[] = {&_binary_swiglu_o_end, &_binary_residual_rmsnorm_o_end,
                           &_binary_linear_o_end};
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
