// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <climits>
#include <cstring>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_reduce_compensated_o_start, _binary_reduce_compensated_o_end;
namespace {
constexpr const char* kName = "gk_block128_neumaier_bf16_compensated_v1";
template <class T>
void map(T& pattern, int dim, int index, float stride, int lo, int hi) {
  pattern.mapping[dim].indexSpaceDim = index;
  pattern.mapping[dim].a = stride;
  pattern.mapping[dim].start_b = lo;
  pattern.mapping[dim].end_b = hi;
}
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  const unsigned capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
  if (info && capacity && *count) std::strcpy(info[0].name, kName);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                 HabanaKernelInstantiation* out) {
  if (std::strcmp(in->guid.name, kName)) return GLUE_NODE_NOT_FOUND;
  if (in->inputTensorNr != 4 || in->outputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const auto& x = in->inputTensors[0].geometry;
  const auto& a = in->inputTensors[1].geometry;
  const auto& w = in->inputTensors[2].geometry;
  const auto& b = in->inputTensors[3].geometry;
  const auto& y = in->outputTensors[0].geometry;
  for (unsigned i = 0; i < 4; ++i) {
    if (in->inputTensors[i].geometry.dataType != DATA_F32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->inputTensorAccessPattern[i].allRequired = 0;
    for (unsigned d = 0; d < in->inputTensors[i].geometry.dims; ++d)
      if (!in->inputTensors[i].geometry.maxSizes[d] ||
          in->inputTensors[i].geometry.maxSizes[d] > INT32_MAX - 128)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
  }
  if (x.dims != 3 || y.dims != 2 || y.dataType != DATA_BF16 ||
      x.maxSizes[0] != y.maxSizes[0] || x.maxSizes[1] != y.maxSizes[1] ||
      a.dims != 3 || a.maxSizes[0] != 1 || a.maxSizes[1] != x.maxSizes[1] ||
      a.maxSizes[2] != x.maxSizes[2] || w.dims != 2 || w.maxSizes[0] != x.maxSizes[2] ||
      w.maxSizes[1] != (x.maxSizes[0] + 127) / 128 ||
      b.dims != 1 || b.maxSizes[0] != x.maxSizes[0])
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->outputTensorAccessPattern[0].allRequired = 0;
  const unsigned groups = x.maxSizes[2];
  out->indexSpaceRank = 2;
  out->indexSpaceGeometry[0] = (y.maxSizes[0] + 127) / 128;
  out->indexSpaceGeometry[1] = y.maxSizes[1];
  auto& xp = out->inputTensorAccessPattern[0];
  map(xp, 0, 0, 128, 0, 127); map(xp, 1, 1, 1, 0, 0); map(xp, 2, 0, 0, 0, groups - 1);
  auto& ap = out->inputTensorAccessPattern[1];
  map(ap, 0, 0, 0, 0, 0); map(ap, 1, 1, 1, 0, 0); map(ap, 2, 0, 0, 0, groups - 1);
  auto& wp = out->inputTensorAccessPattern[2];
  map(wp, 0, 0, 0, 0, groups - 1); map(wp, 1, 0, 1, 0, 0);
  auto& bp = out->inputTensorAccessPattern[3];
  map(bp, 0, 0, 128, 0, 127);
  auto& yp = out->outputTensorAccessPattern[0];
  map(yp, 0, 0, 128, 0, 127); map(yp, 1, 1, 1, 0, 0);
  const auto bytes = static_cast<unsigned>(&_binary_reduce_compensated_o_end -
                                           &_binary_reduce_compensated_o_start);
  const unsigned capacity = out->kernel.elfSize;
  out->kernel.paramsNr = 0;
  out->kernel.elfSize = bytes;
  if (capacity < bytes) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf, &_binary_reduce_compensated_o_start, bytes);
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* in,
                                              ShapeInferenceOutput*) {
  return std::strcmp(in->pGuid ? in->pGuid->name : in->guid.name, kName)
             ? GLUE_NODE_NOT_FOUND : GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (std::strcmp(in->guid.name, kName)) return GLUE_NODE_NOT_FOUND;
  const unsigned capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < in->inputTensorNr; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < in->outputTensorNr; ++i)
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
