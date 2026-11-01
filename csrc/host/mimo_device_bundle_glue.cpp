// SPDX-License-Identifier: Apache-2.0
// Synapse limits the number of TPC libraries. Expose the two model-owned
// providers through one library without changing either kernel implementation.
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>

using namespace tpc_lib_api;
extern "C" GlueCodeReturn full_InstantiateTpcKernel(HabanaKernelParams*, HabanaKernelInstantiation*);
extern "C" GlueCodeReturn tp8_InstantiateTpcKernel(HabanaKernelParams*, HabanaKernelInstantiation*);
extern "C" GlueCodeReturn full_GetShapeInference(DeviceId, const ShapeInferenceParams*, ShapeInferenceOutput*);
extern "C" GlueCodeReturn tp8_GetShapeInference(DeviceId, const ShapeInferenceParams*, ShapeInferenceOutput*);
extern "C" GlueCodeReturn full_GetSupportedDataLayout(const HabanaKernelParams*, NodeDataLayouts*, uint32_t*);
extern "C" GlueCodeReturn tp8_GetSupportedDataLayout(const HabanaKernelParams*, NodeDataLayouts*, uint32_t*);

static const char* names[] = {"gk_full_mme_gather_v1", "gk_full_mme_softmax_v2",
                              "pf_sum_bf16_bf16", "pf_sum_f32_bf16"};
static int kind(const char* name) {
  for (int i = 0; i < 4; ++i) if (!std::strcmp(name, names[i])) return i;
  return -1;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  const unsigned capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 4 : 0;
  if (info) for (unsigned i = 0; i < std::min(capacity, *count); ++i) std::strcpy(info[i].name, names[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                HabanaKernelInstantiation* out) {
  const int id = kind(in->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  return id < 2 ? full_InstantiateTpcKernel(in, out) : tp8_InstantiateTpcKernel(in, out);
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* in,
                                             ShapeInferenceOutput* out) {
  const int id = kind(in->pGuid ? in->pGuid->name : in->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  return id < 2 ? full_GetShapeInference(device, in, out) : tp8_GetShapeInference(device, in, out);
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,
                                                  NodeDataLayouts* out, uint32_t* count) {
  const int id = kind(in->guid.name);
  if (id < 0) return GLUE_NODE_NOT_FOUND;
  return id < 2 ? full_GetSupportedDataLayout(in, out, count) : tp8_GetSupportedDataLayout(in, out, count);
}
extern "C" uint64_t GetLibVersion() { return 1; }
