#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_candidate_o_start,_binary_candidate_o_end;
static const char*name="gk_moe_gp_scale_tail_v1";
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?1:0;if(g&&cap&&*n)std::strcpy(g[0].name,name);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 if(std::strcmp(in->guid.name,name))return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=5||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto&x=in->inputTensors[2].geometry;auto&ids=in->inputTensors[4].geometry;auto&y=in->outputTensors[0].geometry;
 auto topk=ids.maxSizes[0],tokens=ids.maxSizes[1],slots=topk*tokens;
 if(x.dataType!=DATA_BF16||ids.dataType!=DATA_I32||y.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_DATA_TYPE;
 if(ids.dims!=2||x.dims!=2||y.dims!=2||!topk||topk>384||!tokens||tokens>INT_MAX/384||slots>INT_MAX/1536)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(x.maxSizes[0]!=6144||x.maxSizes[1]!=tokens||y.maxSizes[0]!=slots*1536||y.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<5;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&a=out->inputTensorAccessPattern[2];a.allRequired=0;a.mapping[0]={0,0,0,6143};a.mapping[1]={0,1.f/(3*topk),-1,1};
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=slots*3;auto&p=out->outputTensorAccessPattern[0];p.mapping[0]={0,512,0,511};p.mapping[1]={0,0,0,0};
 out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/topk);
 unsigned size=&_binary_candidate_o_end-&_binary_candidate_o_start,capacity=out->kernel.elfSize;out->kernel.elfSize=size;if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,&_binary_candidate_o_start,size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return std::strcmp(in->pGuid?in->pGuid->name:in->guid.name,name)?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(std::strcmp(in->guid.name,name))return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
