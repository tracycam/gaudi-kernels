// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_affine_o_start,_binary_affine_o_end;
static const char*guid="gk_native_graph_affine_f32_v1";
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){auto cap=*n;*n=d==DEVICE_ID_GAUDI2?1:0;if(g&&cap>=*n&&*n)std::strcpy(g[0].name,guid);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*p,HabanaKernelInstantiation*q){
 if(std::strcmp(p->guid.name,guid))return GLUE_NODE_NOT_FOUND;
 if(p->inputTensorNr!=1||p->outputTensorNr!=1||p->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto&x=p->inputTensors[0].geometry;auto&y=p->outputTensors[0].geometry;
 if(x.dims!=2||y.dims!=2||x.dataType!=DATA_F32||y.dataType!=DATA_F32||!x.maxSizes[0]||x.maxSizes[0]>(1u<<24)||x.maxSizes[0]%64||x.maxSizes[0]!=y.maxSizes[0]||x.maxSizes[1]!=1||y.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 q->indexSpaceRank=1;q->indexSpaceGeometry[0]=x.maxSizes[0]/64;
 q->inputTensorAccessPattern[0].mapping[0]={0,64,0,63};q->inputTensorAccessPattern[0].mapping[1]={0,0,0,0};
 q->outputTensorAccessPattern[0]=q->inputTensorAccessPattern[0];q->kernel.paramsNr=1;std::memcpy(q->kernel.scalarParams,p->nodeParams.nodeParams,4);
 auto n=&_binary_affine_o_end-&_binary_affine_o_start;auto cap=q->kernel.elfSize;q->kernel.elfSize=n;
 if(cap<n)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(q->kernel.kernelElf,&_binary_affine_o_start,n);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*p,ShapeInferenceOutput*){return std::strcmp(p->pGuid?p->pGuid->name:p->guid.name,guid)?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*p,NodeDataLayouts*l,uint32_t*c){if(std::strcmp(p->guid.name,guid))return GLUE_NODE_NOT_FOUND;auto cap=*c;*c=1;if(l&&cap){std::memset(l[0].inputs[0].layout,'x',sizeof(l[0].inputs[0].layout));std::memset(l[0].outputs[0].layout,'x',sizeof(l[0].outputs[0].layout));}return GLUE_SUCCESS;}
