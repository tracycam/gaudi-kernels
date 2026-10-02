#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_stream_decode_o_start,_binary_stream_decode_o_end;
static const char* guid="gk_moe_stream_decode_historical_k8";
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?1:0;if(g&&*n)std::strcpy(g[0].name,guid);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 if(std::strcmp(in->guid.name,guid))return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=4||in->outputTensorNr!=1||in->nodeParams.nodeParamsSize!=12)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto&y=in->outputTensors[0].geometry;
 if(y.dims!=3||y.maxSizes[0]%512||y.maxSizes[1]%32||y.maxSizes[2]>2||y.maxSizes[0]*y.maxSizes[1]*y.maxSizes[2]*2>16*1024*1024)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,12);
 for(int i=0;i<4;++i)out->inputTensorAccessPattern[i].allRequired=1;
 out->indexSpaceRank=3;out->indexSpaceGeometry[0]=y.maxSizes[0]/256;out->indexSpaceGeometry[1]=y.maxSizes[1]/32;out->indexSpaceGeometry[2]=y.maxSizes[2];
 auto&o=out->outputTensorAccessPattern[0];map(o,0,0,256,0,255);map(o,1,1,32,0,31);map(o,2,2,1,0,0);
 unsigned size=&_binary_stream_decode_o_end-&_binary_stream_decode_o_start,cap=out->kernel.elfSize;out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,&_binary_stream_decode_o_start,size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return std::strcmp(in->pGuid?in->pGuid->name:in->guid.name,guid)?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(std::strcmp(in->guid.name,guid))return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
