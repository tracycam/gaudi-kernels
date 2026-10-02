#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_control_o_start,_binary_control_o_end;
extern "C" unsigned char _binary_candidate_o_start,_binary_candidate_o_end;
extern "C" unsigned char _binary_unroll8_o_start,_binary_unroll8_o_end;
namespace {
const char*names[]={"gk_moe_combine_scalar_control_experiment","gk_moe_combine_vector_routes_experiment","gk_moe_combine_vector_routes_unroll8_experiment"};
int which(const char*n){for(int i=0;i<3;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
bool shape(const TensorGeometry&g,TensorDataType type,uint64_t x,uint64_t y){
 if(g.dataType!=type||g.dims!=2||g.maxSizes[0]!=x||g.maxSizes[1]!=y)return false;
 return g.minSizes[0]==x&&g.minSizes[1]==y;
}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?3:0;if(g)for(unsigned i=0;i<*n&&i<cap;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->deviceId!=DEVICE_ID_GAUDI2)return GLUE_FAILED;
 if(in->inputTensorNr!=3||in->outputTensorNr!=1||in->nodeParams.nodeParamsSize)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto&y=in->outputTensors[0].geometry;uint64_t h=y.maxSizes[0],t=y.maxSizes[1];
 if(!h||h>6144||h%128||!t||t>8||!shape(y,DATA_F32,h,t)
    ||!shape(in->inputTensors[0].geometry,DATA_F32,t*8*h,1)
    ||!shape(in->inputTensors[1].geometry,DATA_F32,8,t)
    ||!shape(in->inputTensors[2].geometry,DATA_U8,256,2))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 // Keep production's index space and access patterns for a controlled A/B.
 for(int i=0;i<3;++i)out->inputTensorAccessPattern[i].allRequired=1;
 out->outputTensorAccessPattern[0].allRequired=1;
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=t*h/128;
 out->kernel.paramsNr=1;out->kernel.scalarParams[0]=8;
 unsigned char*starts[]={&_binary_control_o_start,&_binary_candidate_o_start,&_binary_unroll8_o_start};
 unsigned char*ends[]={&_binary_control_o_end,&_binary_candidate_o_end,&_binary_unroll8_o_end};
 unsigned char*start=starts[id],*end=ends[id];
 unsigned size=end-start,cap=out->kernel.elfSize;out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start,size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
