// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_swiglu_o_start,_binary_swiglu_o_end;
extern "C" unsigned char _binary_residual_rmsnorm_o_start,_binary_residual_rmsnorm_o_end;
extern "C" unsigned char _binary_linear_o_start,_binary_linear_o_end;
namespace {
constexpr const char* names[]={"gk_dense_swiglu_prefill_fp32_v1","gk_dense_residual_rmsnorm_prefill_fp32_v1","gk_dense_linear_prefill_fp32_v1"};
int kind(const char* name) { for(int i=0;i<3;++i) if(!std::strcmp(name,names[i])) return i; return -1; }
bool shape(const Tensor& t,TensorDataType type,std::initializer_list<unsigned long> sizes) {
  if(t.geometry.dataType!=type || t.geometry.dims!=sizes.size()) return false;
  unsigned i=0; for(auto size:sizes) { if(t.geometry.maxSizes[i]!=size || t.geometry.minSizes[i]!=size) return false; ++i; }
  return true;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t* count,GuidInfo* info) {
  const unsigned cap=*count; *count=device==DEVICE_ID_GAUDI2?3:0;
  if(info) for(unsigned i=0;i<cap && i<*count;++i) std::strcpy(info[i].name,names[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
  const int variant=kind(in->guid.name); if(variant<0) return GLUE_NODE_NOT_FOUND;
  const unsigned inputs=variant==0?1:(variant==1?3:4),outputs=variant==1?2:1;
  if(in->inputTensorNr!=inputs || in->outputTensorNr!=outputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  out->kernel.paramsNr=0;
  const auto rows=in->inputTensors[0].geometry.maxSizes[1];
  if(rows!=32 && rows!=256 && rows!=512 && rows!=1024 && rows!=2048) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if(variant==0) {
    if(!shape(in->inputTensors[0],DATA_F32,{4096,rows}) || !shape(in->outputTensors[0],DATA_F32,{2048,rows})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank=2; out->indexSpaceGeometry[0]=16; out->indexSpaceGeometry[1]=rows;
  } else if(variant==1) {
    if(!shape(in->inputTensors[0],DATA_BF16,{6144,rows}) || !shape(in->inputTensors[1],DATA_BF16,{6144,rows}) ||
       !shape(in->inputTensors[2],DATA_BF16,{6144}) || !shape(in->outputTensors[0],DATA_BF16,{6144,rows}) ||
       !shape(in->outputTensors[1],DATA_F32,{6144,rows}) || in->nodeParams.nodeParamsSize!=sizeof(float) || !in->nodeParams.nodeParams)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const float eps=*static_cast<const float*>(in->nodeParams.nodeParams);
    if(!std::isfinite(eps) || eps<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    out->indexSpaceRank=1; out->indexSpaceGeometry[0]=rows; out->kernel.paramsNr=1; std::memcpy(out->kernel.scalarParams,&eps,sizeof(eps));
  } else {
    const auto& w=in->inputTensors[1].geometry;
    if(w.dims!=3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned n=w.maxSizes[1],groups=w.maxSizes[2];
    if((n!=4096 && n!=6144) || !groups || !shape(in->inputTensors[0],DATA_F32,{groups*128,rows}) ||
       !shape(in->inputTensors[1],DATA_F8_143,{128,n,groups}) || !shape(in->inputTensors[2],DATA_F32,{groups,n/128}) ||
       !shape(in->inputTensors[3],DATA_F32,{n}) || !shape(in->outputTensors[0],DATA_F32,{n,rows})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank=2; out->indexSpaceGeometry[0]=n/128; out->indexSpaceGeometry[1]=rows;
  }
  for(unsigned i=0;i<inputs;++i) out->inputTensorAccessPattern[i].allRequired=1;
  for(unsigned i=0;i<outputs;++i) out->outputTensorAccessPattern[i].allRequired=1;
  unsigned char* starts[]={&_binary_swiglu_o_start,&_binary_residual_rmsnorm_o_start,&_binary_linear_o_start};
  unsigned char* ends[]={&_binary_swiglu_o_end,&_binary_residual_rmsnorm_o_end,&_binary_linear_o_end};
  const unsigned cap=out->kernel.elfSize; out->kernel.elfSize=ends[variant]-starts[variant];
  if(cap<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf,starts[variant],out->kernel.elfSize); return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){ return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*) {
  return kind(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count) {
  if(kind(in->guid.name)<0) return GLUE_NODE_NOT_FOUND;
  const unsigned cap=*count; *count=1; if(!layouts || !cap) return GLUE_SUCCESS;
  for(unsigned i=0;i<in->inputTensorNr;++i) std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
  for(unsigned i=0;i<in->outputTensorNr;++i) std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
