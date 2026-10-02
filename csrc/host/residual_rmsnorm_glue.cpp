// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
#include <climits>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_fused_o_start,_binary_fused_o_end,_binary_per_row_o_start,_binary_per_row_o_end,_binary_block128_o_start,_binary_block128_o_end;
extern "C" unsigned char _binary_pure_o_start,_binary_pure_o_end;
static const char* guids[]={"gk_residual_rmsnorm_bf16_v1","gk_residual_rmsnorm_per_row_a8_v1","gk_residual_rmsnorm_block128_a8_v1","gk_pure_rmsnorm_bf16_v1"};
static int which(const char*n){for(int i=0;i<4;++i)if(!std::strcmp(n,guids[i]))return i;return -1;}
static bool known(const char*n){return which(n)>=0;}
template<class T> void map(T&p,int d,int index,float step,int hi){p.mapping[d].indexSpaceDim=index;p.mapping[d].a=step;p.mapping[d].start_b=0;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*c,GuidInfo*g){*c=d==DEVICE_ID_GAUDI2?4:0;if(g)for(unsigned i=0;i<*c;++i)std::strcpy(g[i].name,guids[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int kind=which(in->guid.name);if(kind<0)return GLUE_NODE_NOT_FOUND;
 if(kind==3){
  if(in->inputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if(in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  float eps=*static_cast<const float*>(in->nodeParams.nodeParams);if(!std::isfinite(eps)||eps<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto&x=in->inputTensors[0].geometry;const auto&w=in->inputTensors[1].geometry;const auto&y=in->outputTensors[0].geometry;
  if(x.dims!=2||!x.maxSizes[0]||x.maxSizes[0]>8192||!x.maxSizes[1]||x.maxSizes[1]>INT_MAX||w.dims!=1||w.maxSizes[0]!=x.maxSizes[0])return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if(y.dims!=2||y.maxSizes[0]!=x.maxSizes[0]||y.maxSizes[1]!=x.maxSizes[1])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  for(const auto*t:{&x,&w,&y}){if(t->dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;for(unsigned d=0;d<t->dims;++d)if(t->maxSizes[d]!=t->minSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;}
  out->indexSpaceRank=1;out->indexSpaceGeometry[0]=x.maxSizes[1];
  for(unsigned i=0;i<2;++i){auto&p=out->inputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,x.maxSizes[0]-1);if(i==0)map(p,1,0,1,0);}
  auto&p=out->outputTensorAccessPattern[0];p.allRequired=0;map(p,0,0,0,x.maxSizes[0]-1);map(p,1,0,1,0);
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&eps,4);unsigned cap=out->kernel.elfSize;out->kernel.elfSize=&_binary_pure_o_end-&_binary_pure_o_start;
  if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,&_binary_pure_o_start,out->kernel.elfSize);return GLUE_SUCCESS;
 }
 if(in->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->outputTensorNr!=(kind?3u:2u))return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float eps=*static_cast<const float*>(in->nodeParams.nodeParams);if(!std::isfinite(eps)||eps<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 const auto& x=in->inputTensors[0].geometry;
 if(x.dims!=2||!x.maxSizes[0]||x.maxSizes[0]>8192||!x.maxSizes[1]||x.maxSizes[1]>INT_MAX)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<3;++i){const auto&t=in->inputTensors[i].geometry;
  if(t.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=(i==2?1u:2u)||t.maxSizes[0]!=x.maxSizes[0]||(i!=2&&t.maxSizes[1]!=x.maxSizes[1]))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 for(unsigned i=0;i<in->outputTensorNr;++i){const auto&t=in->outputTensors[i].geometry;
  if(t.dataType!=(i==0||!kind?DATA_BF16:(i==1?DATA_F8_143:DATA_F32)))return GLUE_INCOMPATIBLE_DATA_TYPE;
  unsigned rank=(kind==2&&i>0)?3:2;unsigned fcd=i==0||!kind?x.maxSizes[0]:(i==2?1:(kind==2?128:x.maxSizes[0]));
  if(t.dims!=rank||t.maxSizes[0]!=fcd||t.maxSizes[1]!=x.maxSizes[1]||(rank==3&&t.maxSizes[2]!=(x.maxSizes[0]+127)/128))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 }
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=x.maxSizes[1];
 for(unsigned i=0;i<3;++i){auto&p=out->inputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,x.maxSizes[0]-1);if(i!=2)map(p,1,0,1,0);}
 for(unsigned i=0;i<in->outputTensorNr;++i){auto&p=out->outputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,in->outputTensors[i].geometry.maxSizes[0]-1);map(p,1,0,1,0);if(kind==2&&i>0)map(p,2,0,0,(x.maxSizes[0]+127)/128-1);}
 unsigned char*start[]={&_binary_fused_o_start,&_binary_per_row_o_start,&_binary_block128_o_start};
 unsigned char*end[]={&_binary_fused_o_end,&_binary_per_row_o_end,&_binary_block128_o_end};
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&eps,4);
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end[kind]-start[kind];
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start[kind],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return known(in->pGuid?in->pGuid->name:in->guid.name)?GLUE_SUCCESS:GLUE_NODE_NOT_FOUND;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*c){if(!known(in->guid.name))return GLUE_NODE_NOT_FOUND;auto cap=*c;*c=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
