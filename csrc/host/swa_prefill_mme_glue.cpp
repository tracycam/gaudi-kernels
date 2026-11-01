// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_swa_mme_gather_o_start,_binary_swa_mme_gather_o_end;
extern "C" unsigned char _binary_swa_mme_softmax_o_start,_binary_swa_mme_softmax_o_end;
namespace {
constexpr const char* names[]={"gk_swa_mme_gather_v1","gk_swa_mme_softmax_v1"};
int which(const char* s){for(int i=0;i<2;++i)if(!std::strcmp(s,names[i]))return i;return -1;}
bool shape(const Tensor& t,TensorDataType type,std::initializer_list<unsigned long> sizes){
 if(t.geometry.dataType!=type||t.geometry.dims!=sizes.size())return false;
 unsigned d=0;for(auto n:sizes){if(t.geometry.minSizes[d]!=n||t.geometry.maxSizes[d]!=n)return false;++d;}return true;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t* count,GuidInfo* out){
 unsigned cap=*count;*count=d==DEVICE_ID_GAUDI2?2:0;
 for(unsigned i=0;out&&i<*count&&i<cap;++i)std::strcpy(out[i].name,names[i]);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=(id?3u:5u)||in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=(id?4u:0u)||(id&&!in->nodeParams.nodeParams))return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 auto*x=in->inputTensors;auto*y=in->outputTensors;
 if(id==0){
  unsigned long slots=x[0].geometry.maxSizes[2];
  if(!slots||slots%128||!shape(x[0],DATA_BF16,{192,1,slots})||!shape(x[1],DATA_BF16,{128,1,slots})||
     !shape(x[2],DATA_I32,{1024})||!shape(x[3],DATA_I32,{1024})||!shape(x[4],DATA_I32,{512})||
     !shape(y[0],DATA_BF16,{192,384,4})||!shape(y[1],DATA_BF16,{128,384,4}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=384;out->indexSpaceGeometry[1]=4;
  for(unsigned i=0;i<2;++i){auto&p=out->outputTensorAccessPattern[i];p.mapping[0]={0,0,0,i?127:191};p.mapping[1]={0,1,0,0};p.mapping[2]={1,1,0,0};}
 }else{
  float scale=*static_cast<float*>(in->nodeParams.nodeParams);
  if(!std::isfinite(scale)||scale<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  if(!shape(x[0],DATA_F32,{384,2048,4})||!shape(x[1],DATA_I32,{512})||!shape(x[2],DATA_BF16,{16})||
     !shape(y[0],DATA_BF16,{384,2048,4})||!shape(y[1],DATA_BF16,{384,2048,4}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=2048;out->indexSpaceGeometry[1]=4;
  for(unsigned i=0;i<2;++i){auto&p=out->outputTensorAccessPattern[i];p.mapping[0]={0,0,0,383};p.mapping[1]={0,1,0,0};p.mapping[2]={1,1,0,0};}
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&scale,4);
 }
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto* start=id?&_binary_swa_mme_softmax_o_start:&_binary_swa_mme_gather_o_start;
 auto* end=id?&_binary_swa_mme_softmax_o_end:&_binary_swa_mme_gather_o_end;
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count){
 if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*count;*count=1;if(!layouts||!cap)return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout));return GLUE_SUCCESS;
}
