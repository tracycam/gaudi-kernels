// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_grid24_o_start,_binary_grid24_o_end;
static const char* guid="gk_norm_block128_grid24_v1";
template<class T>void map(T&p,int d,float a,int lo,int hi){p.mapping[d].indexSpaceDim=0;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
static bool shape(const Tensor&t,TensorDataType type,std::initializer_list<unsigned>sizes){
 if(t.geometry.dataType!=type||t.geometry.dims!=sizes.size())return false;
 unsigned d=0;for(auto s:sizes){if(t.geometry.minSizes[d]!=s||t.geometry.maxSizes[d]!=s)return false;++d;}return true;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?1:0;if(g&&cap&&*n)std::strcpy(g[0].name,guid);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 if(std::strcmp(in->guid.name,guid))return GLUE_NODE_NOT_FOUND;
 if(in->deviceId!=DEVICE_ID_GAUDI2)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 if(in->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(!shape(in->inputTensors[0],DATA_BF16,{6144,1})||!shape(in->inputTensors[1],DATA_BF16,{6144,1})||!shape(in->inputTensors[2],DATA_BF16,{6144}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!shape(in->outputTensors[0],DATA_BF16,{6144,1})||!shape(in->outputTensors[1],DATA_F8_143,{128,1,48})||!shape(in->outputTensors[2],DATA_F32,{1,1,48}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float eps=*static_cast<float*>(in->nodeParams.nodeParams);if(!std::isfinite(eps)||eps<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=24;
 for(unsigned i=0;i<3;++i){auto&p=out->inputTensorAccessPattern[i];p.allRequired=0;map(p,0,i==2?256:0,0,i==2?255:6143);if(i<2)map(p,1,0,0,0);}
 auto&r=out->outputTensorAccessPattern[0];r.allRequired=0;map(r,0,256,0,255);map(r,1,0,0,0);
 for(unsigned i=1;i<3;++i){auto&p=out->outputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,0,i==1?127:0);map(p,1,0,0,0);map(p,2,2,0,1);}
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&eps,4);
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=&_binary_grid24_o_end-&_binary_grid24_o_start;
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,&_binary_grid24_o_start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return std::strcmp(in->pGuid?in->pGuid->name:in->guid.name,guid)?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(std::strcmp(in->guid.name,guid))return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<3;++i){std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
