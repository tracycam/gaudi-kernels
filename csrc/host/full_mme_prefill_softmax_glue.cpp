// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_prefill_softmax_o_start, _binary_prefill_softmax_o_end;
namespace {
constexpr const char* guid = "gk_full_mme_prefill_softmax_v1";
bool shape(const Tensor& t, TensorDataType type, std::initializer_list<unsigned long> sizes) {
  if (t.geometry.dataType != type || t.geometry.dims != sizes.size()) return false;
  unsigned i=0; for (auto size:sizes) {
    if (t.geometry.maxSizes[i]!=size || t.geometry.minSizes[i]!=size) return false;
    ++i;
  }
  return true;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d, uint32_t* count, GuidInfo* info) {
  const unsigned cap=*count; *count=d==DEVICE_ID_GAUDI2 ? 1 : 0;
  if (info && cap && *count) std::strcpy(info[0].name,guid);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
  if (std::strcmp(in->guid.name,guid)) return GLUE_NODE_NOT_FOUND;
  if (in->inputTensorNr!=5 || in->outputTensorNr!=2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  struct Params { float scale; int physical_pages; };
  const unsigned long pages=in->inputTensors[1].geometry.maxSizes[0];
  if (!pages || pages>64 || in->nodeParams.nodeParamsSize!=sizeof(Params) || !in->nodeParams.nodeParams)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const auto& p=*static_cast<const Params*>(in->nodeParams.nodeParams);
  if (!std::isfinite(p.scale) || p.scale<=0 || p.physical_pages<=0)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto rows=in->inputTensors[4].geometry.maxSizes[0];
  if ((rows!=32 && rows!=256 && rows!=512 && rows!=1024) ||
      !shape(in->inputTensors[0],DATA_F32,{pages*128,16*rows}) ||
      !shape(in->inputTensors[1],DATA_I32,{pages}) ||
      !shape(in->inputTensors[2],DATA_I32,{pages}) ||
      !shape(in->inputTensors[3],DATA_I32,{pages}) ||
      !shape(in->inputTensors[4],DATA_I32,{rows}) ||
      !shape(in->outputTensors[0],DATA_BF16,{pages*128,16*rows}) ||
      !shape(in->outputTensors[1],DATA_BF16,{pages*128,16*rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=1; out->indexSpaceGeometry[0]=16*rows;
  for (unsigned i=0;i<in->inputTensorNr;++i) out->inputTensorAccessPattern[i].allRequired=1;
  for (unsigned i=0;i<in->outputTensorNr;++i) out->outputTensorAccessPattern[i].allRequired=1;
  out->kernel.paramsNr=2; std::memcpy(out->kernel.scalarParams,&p,sizeof(p));
  const unsigned capacity=out->kernel.elfSize;
  out->kernel.elfSize=&_binary_prefill_softmax_o_end-&_binary_prefill_softmax_o_start;
  if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf,&_binary_prefill_softmax_o_start,out->kernel.elfSize);
  return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*) {
  return std::strcmp(in->pGuid ? in->pGuid->name : in->guid.name,guid) ? GLUE_NODE_NOT_FOUND : GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count) {
  if (std::strcmp(in->guid.name,guid)) return GLUE_NODE_NOT_FOUND;
  const unsigned cap=*count; *count=1; if (!layouts || !cap) return GLUE_SUCCESS;
  for (unsigned i=0;i<in->inputTensorNr;++i) std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
  for (unsigned i=0;i<in->outputTensorNr;++i) std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
