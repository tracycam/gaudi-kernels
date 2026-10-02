// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
#include <climits>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_head_batch_o_start,_binary_head_batch_o_end;
extern "C" unsigned char _binary_head_batch_quad_o_start,_binary_head_batch_quad_o_end;
static const char* guid="gk_swa128_batch_window_fp32_v1";
static const char* quad_guid="gk_swa128_batch_window_quad_fp32_v2";
static bool known(const char*n){return !std::strcmp(n,guid)||!std::strcmp(n,quad_guid);}
static bool shape(const Tensor&t,TensorDataType type,std::initializer_list<unsigned long>d){if(t.geometry.dataType!=type||t.geometry.dims!=d.size())return false;unsigned i=0;for(auto n:d){if(t.geometry.maxSizes[i]!=n||t.geometry.minSizes[i]!=n)return false;++i;}return true;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*c,GuidInfo*g){*c=d==DEVICE_ID_GAUDI2?2:0;if(g&&*c){std::strcpy(g[0].name,guid);std::strcpy(g[1].name,quad_guid);}return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 if(!known(in->guid.name))return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=7)return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float scale=*static_cast<float*>(in->nodeParams.nodeParams);if(!std::isfinite(scale)||scale<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 unsigned long rows=in->inputTensors[0].geometry.maxSizes[2],slots=in->inputTensors[1].geometry.maxSizes[2],pages=in->inputTensors[3].geometry.maxSizes[0];
 if(!rows||rows>32||!slots||slots%128||slots>INT_MAX/192||!pages||pages>4096)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!shape(in->inputTensors[0],DATA_BF16,{3072,1,rows})||!shape(in->inputTensors[1],DATA_BF16,{192,1,slots})||!shape(in->inputTensors[2],DATA_BF16,{128,1,slots})||!shape(in->inputTensors[6],DATA_BF16,{16})||!shape(in->inputTensors[3],DATA_I32,{pages})||!shape(in->inputTensors[4],DATA_I32,{pages})||!(shape(in->inputTensors[5],DATA_I32,{rows})||shape(in->inputTensors[5],DATA_I32,{1,rows})))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!shape(in->outputTensors[0],DATA_BF16,{128,16,rows}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 out->indexSpaceRank=2;out->indexSpaceGeometry[0]=16;out->indexSpaceGeometry[1]=rows;
 for(unsigned i=0;i<7;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&p=out->outputTensorAccessPattern[0];p.allRequired=0;p.mapping[0]={0,0,0,127};p.mapping[1]={0,1,0,0};p.mapping[2]={1,1,0,0};
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&scale,4);bool quad=!std::strcmp(in->guid.name,quad_guid);auto*start=quad?&_binary_head_batch_quad_o_start:&_binary_head_batch_o_start;auto*end=quad?&_binary_head_batch_quad_o_end:&_binary_head_batch_o_end;unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return known(in->pGuid?in->pGuid->name:in->guid.name)?GLUE_SUCCESS:GLUE_NODE_NOT_FOUND;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*c){if(!known(in->guid.name))return GLUE_NODE_NOT_FOUND;auto cap=*c;*c=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));std::memset(l[0].outputs[0].layout,'x',sizeof(l[0].outputs[0].layout));return GLUE_SUCCESS;}
