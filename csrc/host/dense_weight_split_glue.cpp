// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_dense_weight_split_o_start, _binary_dense_weight_split_o_end;
namespace {
constexpr const char* name = "gk_dense_block128_decode_split_v1";
bool named(const char* value) { return value && !std::strcmp(value,name); }
template<class T> void map(T& p,int dim,int index,float stride,int lo,int hi) {
  p.mapping[dim].indexSpaceDim=index; p.mapping[dim].a=stride;
  p.mapping[dim].start_b=lo; p.mapping[dim].end_b=hi;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t* count,GuidInfo* info) {
  const auto capacity=*count; *count=device==DEVICE_ID_GAUDI2?1:0;
  if(info && capacity && *count) std::strcpy(info[0].name,name);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
  if(!named(in->guid.name)) return GLUE_NODE_NOT_FOUND;
  if(in->inputTensorNr!=2 || in->outputTensorNr!=2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const auto& w=in->inputTensors[0].geometry; const auto& s=in->inputTensors[1].geometry;
  const auto n=w.maxSizes[1], groups=w.maxSizes[2];
  if(w.dims!=3 || w.dataType!=DATA_F8_143 || w.maxSizes[0]!=128 || !n || n%128 || !groups ||
     s.dims!=2 || s.dataType!=DATA_F32 || s.maxSizes[0]!=groups || s.maxSizes[1]!=n/128)
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned i=0;i<2;++i) {
    const auto& y=in->outputTensors[i].geometry;
    if(y.dims!=2 || y.dataType!=DATA_BF16 || y.maxSizes[0]!=128*groups || y.maxSizes[1]!=n)
      return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->outputTensorAccessPattern[i].allRequired=0;
    map(out->outputTensorAccessPattern[i],0,0,128,0,127);
    map(out->outputTensorAccessPattern[i],1,1,128,0,127);
  }
  out->indexSpaceRank=2; out->indexSpaceGeometry[0]=groups; out->indexSpaceGeometry[1]=n/128;
  auto& a=out->inputTensorAccessPattern[0]; a.allRequired=0;
  map(a,0,0,0,0,127); map(a,1,1,128,0,127); map(a,2,0,1,0,0);
  auto& b=out->inputTensorAccessPattern[1]; b.allRequired=0;
  map(b,0,0,1,0,0); map(b,1,1,1,0,0);
  out->kernel.paramsNr=0;
  const auto size=&_binary_dense_weight_split_o_end-&_binary_dense_weight_split_o_start;
  const auto capacity=out->kernel.elfSize; out->kernel.elfSize=size;
  if(capacity<size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf,&_binary_dense_weight_split_o_start,size);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*) {
  return named(in->pGuid?in->pGuid->name:in->guid.name)?GLUE_SUCCESS:GLUE_NODE_NOT_FOUND;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count) {
  if(!named(in->guid.name)) return GLUE_NODE_NOT_FOUND;
  const auto capacity=*count; *count=1;
  if(!layouts || !capacity) return GLUE_SUCCESS;
  for(unsigned i=0;i<in->inputTensorNr;++i) std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
  for(unsigned i=0;i<in->outputTensorNr;++i) std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion() { return 1; }
