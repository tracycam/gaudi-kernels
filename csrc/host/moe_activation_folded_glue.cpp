#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_folded_gp_o_start,_binary_folded_gp_o_end;
extern "C" unsigned char _binary_folded_down_o_start,_binary_folded_down_o_end;
namespace {
const char*names[]={"gk_moe_gp_folded_v1","gk_moe_down_folded_v1"};
int which(const char*n){for(int i=0;i<2;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t*n,GuidInfo*g){*n=device==DEVICE_ID_GAUDI2?2:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=5||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto&x=in->inputTensors[2].geometry;const auto&ids=in->inputTensors[4].geometry;const auto&y=in->outputTensors[0].geometry;
 if(x.dataType!=DATA_BF16||ids.dataType!=DATA_I32||y.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_DATA_TYPE;
 auto topk=ids.maxSizes[0],tokens=ids.maxSizes[1],slots=topk*tokens;
 if(ids.dims!=2||x.dims!=2||y.dims!=2||topk<1||topk>384||!tokens||tokens>INT_MAX/384||slots>INT_MAX/(id?6144:1536))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(x.maxSizes[0]!=(id?256:6144)||x.maxSizes[1]!=(id?slots:tokens)||y.maxSizes[0]!=slots*(id?6144:1536)||y.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<5;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&a=out->inputTensorAccessPattern[2];a.allRequired=0;
 a.mapping[0]={0,0,0,float(id?255:6143)};
 a.mapping[1]={0,1.f/(id?12:3*topk),-1,1};
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=slots*(id?12:3);
 auto&p=out->outputTensorAccessPattern[0];p.mapping[0]={0,512,0,511};p.mapping[1]={0,0,0,0};
 out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/topk);
 unsigned char*start=id?&_binary_folded_down_o_start:&_binary_folded_gp_o_start;unsigned char*end=id?&_binary_folded_down_o_end:&_binary_folded_gp_o_end;
 unsigned size=end-start,capacity=out->kernel.elfSize;out->kernel.elfSize=size;if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start,size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){
 if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
