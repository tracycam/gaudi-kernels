// SPDX-License-Identifier: Apache-2.0
// Model-owned providers share one Synapse library slot.
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <vector>
using namespace tpc_lib_api;
#define DECLARE(prefix) \
extern "C" GlueCodeReturn prefix##_GetKernelGuids(DeviceId,uint32_t*,GuidInfo*); \
extern "C" GlueCodeReturn prefix##_InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*); \
extern "C" GlueCodeReturn prefix##_GetShapeInference(DeviceId,const ShapeInferenceParams*,ShapeInferenceOutput*); \
extern "C" GlueCodeReturn prefix##_GetSupportedDataLayout(const HabanaKernelParams*,NodeDataLayouts*,uint32_t*);
DECLARE(full) DECLARE(moe) DECLARE(dense) DECLARE(embedding) DECLARE(argmax) DECLARE(qkv)
#undef DECLARE
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t* count,GuidInfo* out) {
  const auto capacity=*count; uint32_t total=0;
  for(auto provider : {full_GetKernelGuids,moe_GetKernelGuids,dense_GetKernelGuids,embedding_GetKernelGuids,argmax_GetKernelGuids,qkv_GetKernelGuids}) {
    uint32_t n=0; auto result=provider(device,&n,nullptr); if(result!=GLUE_SUCCESS) return result;
    std::vector<GuidInfo> guids(n); auto supplied=n; result=provider(device,&supplied,guids.data());
    if(result!=GLUE_SUCCESS) return result;
    for(uint32_t i=0;i<n && total+i<capacity && out;++i) out[total+i]=guids[i];
    total+=n;
  }
  *count=total; return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
  for(auto provider : {full_InstantiateTpcKernel,moe_InstantiateTpcKernel,dense_InstantiateTpcKernel,embedding_InstantiateTpcKernel,argmax_InstantiateTpcKernel,qkv_InstantiateTpcKernel}) {
    auto result=provider(in,out); if(result!=GLUE_NODE_NOT_FOUND) return result;
  }
  return GLUE_NODE_NOT_FOUND;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId device,const ShapeInferenceParams* in,ShapeInferenceOutput* out) {
  for(auto provider : {full_GetShapeInference,moe_GetShapeInference,dense_GetShapeInference,embedding_GetShapeInference,argmax_GetShapeInference,qkv_GetShapeInference}) {
    auto result=provider(device,in,out); if(result!=GLUE_NODE_NOT_FOUND) return result;
  }
  return GLUE_NODE_NOT_FOUND;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* out,uint32_t* count) {
  for(auto provider : {full_GetSupportedDataLayout,moe_GetSupportedDataLayout,dense_GetSupportedDataLayout,embedding_GetSupportedDataLayout,argmax_GetSupportedDataLayout,qkv_GetSupportedDataLayout}) {
    auto result=provider(in,out,count); if(result!=GLUE_NODE_NOT_FOUND) return result;
  }
  return GLUE_NODE_NOT_FOUND;
}
extern "C" uint64_t GetLibVersion() { return 1; }
