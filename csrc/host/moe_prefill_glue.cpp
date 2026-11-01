// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_router_post_o_start,_binary_router_post_o_end;
extern "C" unsigned char _binary_gate_o_start,_binary_gate_o_end;
extern "C" unsigned char _binary_combine_o_start,_binary_combine_o_end;
namespace {
constexpr const char* names[]={"gk_router_post8_prefill_v1","gk_mimo_moe_gate_prefill_fp32_v1","gk_mimo_moe_combine_prefill_fp32_v1"};
int kind(const char* name) { for(int i=0;i<3;++i) if(!std::strcmp(name,names[i])) return i; return -1; }
bool shape(const Tensor& t,TensorDataType type,std::initializer_list<unsigned long> sizes) {
 if(t.geometry.dataType!=type || t.geometry.dims!=sizes.size()) return false;
 unsigned i=0; for(auto size:sizes) { if(t.geometry.maxSizes[i]!=size || t.geometry.minSizes[i]!=size) return false; ++i; } return true;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t* count,GuidInfo* info) {
 const unsigned cap=*count; *count=device==DEVICE_ID_GAUDI2?3:0;
 if(info) for(unsigned i=0;i<cap && i<*count;++i) std::strcpy(info[i].name,names[i]); return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
 const int k=kind(in->guid.name); if(k<0) return GLUE_NODE_NOT_FOUND;
 const unsigned inputs=k==2?3:2,outputs=k==0?3:1;
 if(in->inputTensorNr!=inputs || in->outputTensorNr!=outputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(k==0) {
  struct Params { float factor; int renormalize; int apply_scale; };
  if(in->nodeParams.nodeParamsSize!=sizeof(Params) || !in->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& p=*static_cast<const Params*>(in->nodeParams.nodeParams);
  if(!std::isfinite(p.factor) || (p.renormalize!=0 && p.renormalize!=1) || (p.apply_scale!=0 && p.apply_scale!=1)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto rows=in->inputTensors[0].geometry.maxSizes[1];
  if((rows!=32 && rows!=256 && rows!=512) ||
     !shape(in->inputTensors[0],DATA_F32,{384,rows}) || !shape(in->inputTensors[1],DATA_I32,{8,rows}) ||
     !shape(in->outputTensors[0],DATA_F32,{8,rows}) || !shape(in->outputTensors[1],DATA_I32,{8,rows}) ||
     !shape(in->outputTensors[2],DATA_F32,{1,rows})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=rows; out->kernel.paramsNr=3; std::memcpy(out->kernel.scalarParams,&p,sizeof(p));
 } else {
  if(in->nodeParams.nodeParamsSize!=sizeof(int32_t) || !in->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  int32_t parameter; std::memcpy(&parameter,in->nodeParams.nodeParams,sizeof(parameter));
  if(parameter!=(k==1?3:8)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  if(k==1) {
   if(!shape(in->inputTensors[0],DATA_F32,{8*32*1536,1}) || !shape(in->inputTensors[1],DATA_I32,{8,32}) ||
      !shape(in->outputTensors[0],DATA_BF16,{256,8*32})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
   out->indexSpaceGeometry[0]=8*32;
  } else {
   if(!shape(in->inputTensors[0],DATA_F32,{8*32*6144,1}) || !shape(in->inputTensors[1],DATA_F32,{8,32}) ||
      !shape(in->inputTensors[2],DATA_U8,{256,2}) || !shape(in->outputTensors[0],DATA_F32,{6144,32})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
   out->indexSpaceGeometry[0]=48*32;
  }
  out->kernel.paramsNr=1; std::memcpy(out->kernel.scalarParams,&parameter,sizeof(parameter));
 }
 out->indexSpaceRank=1;
 for(unsigned i=0;i<inputs;++i) out->inputTensorAccessPattern[i].allRequired=1;
 for(unsigned i=0;i<outputs;++i) out->outputTensorAccessPattern[i].allRequired=1;
 unsigned char* starts[]={&_binary_router_post_o_start,&_binary_gate_o_start,&_binary_combine_o_start};
 unsigned char* ends[]={&_binary_router_post_o_end,&_binary_gate_o_end,&_binary_combine_o_end};
 const unsigned cap=out->kernel.elfSize; out->kernel.elfSize=ends[k]-starts[k];
 if(cap<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,starts[k],out->kernel.elfSize); return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){ return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*) {
 return kind(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count) {
 if(kind(in->guid.name)<0) return GLUE_NODE_NOT_FOUND;
 const unsigned cap=*count; *count=1; if(!layouts || !cap) return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i) std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i) std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout)); return GLUE_SUCCESS;
}
