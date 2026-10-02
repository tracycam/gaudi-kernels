// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
#include <climits>
using namespace tpc_lib_api;
#define VARIANTS(X) X(bf16_bf16) X(bf16_f32) X(f32_bf16) X(f32_f32) X(cache_bf16_v2)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
VARIANTS(DECL)
#define NAME(n) "gk_qkv_post_" #n "_v1",
static const char*names[]={"gk_qkv_post_bf16_bf16_v1","gk_qkv_post_bf16_f32_v1","gk_qkv_post_f32_bf16_v1","gk_qkv_post_f32_f32_v1","gk_qkv_post_cache_bf16_v2"};
static int which(const char*n){for(int i=0;i<5;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T> static void map(T&p,int dim,int index,int step,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=step;p.mapping[dim].start_b=0;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*c,GuidInfo*g){*c=d==DEVICE_ID_GAUDI2?5:0;if(g)for(unsigned i=0;i<*c;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float scale=*static_cast<const float*>(in->nodeParams.nodeParams);if(!std::isfinite(scale))return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 bool cached=id==4;
 auto&x=in->inputTensors[0].geometry;unsigned m=x.maxSizes[1];
 if(x.dims!=2||x.maxSizes[0]!=3392||x.maxSizes[1]==0||x.maxSizes[1]>(cached?INT_MAX/3392:INT_MAX))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(cached){
  auto&c=in->inputTensors[1].geometry;auto&p=in->inputTensors[2].geometry;
  if(x.dataType!=DATA_BF16||c.dataType!=DATA_BF16||p.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(c.dims!=2||c.maxSizes[0]!=64||!c.maxSizes[1]||c.maxSizes[1]>INT_MAX/64)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if(!((p.dims==1&&p.maxSizes[0]==m)||(p.dims==2&&p.maxSizes[0]*p.maxSizes[1]==m&&(p.maxSizes[0]==1||p.maxSizes[1]==1))))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned i=0;i<3;++i){auto&t=in->inputTensors[i].geometry;for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;}
  for(unsigned i=0;i<3;++i){auto&t=in->outputTensors[i].geometry;if(t.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
   if(i==0){if(t.dims!=2||t.maxSizes[0]!=3072||t.maxSizes[1]!=m)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;}
   else if(t.dims!=3||t.maxSizes[0]!=(i==2?128u:192u)||t.maxSizes[1]!=1||t.maxSizes[2]!=m)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
   for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  }
 }else{
 for(unsigned i=0;i<3;++i){auto&t=in->inputTensors[i].geometry;if(t.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(i&&(t.dims!=3||t.maxSizes[0]!=64||t.maxSizes[1]!=1||t.maxSizes[2]!=m))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 for(unsigned i=0;i<3;++i){auto&t=in->outputTensors[i].geometry;if(t.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=3||t.maxSizes[0]!=(i==2?128u:192u)||t.maxSizes[1]!=(i==0?16u:1u)||t.maxSizes[2]!=m)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 }
 }
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=m;
 if(cached){
  auto&xp=out->inputTensorAccessPattern[0];xp.allRequired=0;map(xp,0,0,0,3391);map(xp,1,0,1,0);
  out->inputTensorAccessPattern[1].allRequired=1;
  auto&pp=out->inputTensorAccessPattern[2];pp.allRequired=0;bool column=in->inputTensors[2].geometry.maxSizes[0]==1;
  map(pp,0,0,column?0:1,0);if(in->inputTensors[2].geometry.dims==2)map(pp,1,0,column?1:0,0);
  for(unsigned i=0;i<3;++i){auto&p=out->outputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,i==0?3071:i==2?127:191);map(p,1,0,i==0?1:0,0);if(i)map(p,2,0,1,0);}
 }else{
 for(unsigned i=0;i<3;++i){auto&p=out->inputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,i?63:3391);map(p,1,0,i?0:1,0);if(i)map(p,2,0,1,0);}
 for(unsigned i=0;i<3;++i){auto&p=out->outputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,i==2?127:191);map(p,1,0,0,i==0?15:0);map(p,2,0,1,0);}
 }
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&scale,4);
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={VARIANTS(BEGIN)},*end[]={VARIANTS(END)};
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end[id]-begin[id];if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,begin[id],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*c){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;auto cap=*c;*c=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<3;++i){std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
