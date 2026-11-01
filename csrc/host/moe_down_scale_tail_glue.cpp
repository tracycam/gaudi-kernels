#include <tpc_kernel_lib_interface.h>
#include <climits>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_control_o_start,_binary_control_o_end;
extern "C" unsigned char _binary_candidate_o_start,_binary_candidate_o_end;
namespace {
const char*names[]={"downact_direct_down_broadcast","gk_moe_down_289_control_experiment","gk_moe_down_242_scale_tail_experiment"};
int which(const char*n){if(!n)return -1;for(int i=0;i<3;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
bool geometry(const TensorGeometry&g,TensorDataType type,uint64_t width,uint64_t rows){
 if(g.dims!=2||g.dataType!=type||g.maxSizes[0]!=width||g.maxSizes[1]!=rows)return false;
 for(unsigned i=0;i<g.dims;++i)if(g.minSizes[i]!=g.maxSizes[i])return false;
 return true;
}
}
extern "C" GlueCodeReturn GetKernelNames(char**k,unsigned*n,DeviceId device){
 if(device!=DEVICE_ID_GAUDI2){*n=0;return GLUE_INCOMPATIBLE_INPUT_SIZE;}
 if(!k){*n=3;return GLUE_SUCCESS;}
 if(*n<3){*n=3;return GLUE_INSUFFICIENT_ISA_BUFFER;}
 *n=3;for(unsigned i=0;i<3;++i)std::strcpy(k[i],names[i]);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=device==DEVICE_ID_GAUDI2?3:0;if(g)for(unsigned i=0;i<*n&&i<cap;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->deviceId!=DEVICE_ID_GAUDI2)return GLUE_FAILED;
 if(in->inputTensorNr!=5||in->outputTensorNr!=1||in->nodeParams.nodeParamsSize)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto&w=in->inputTensors[0].geometry;const auto&s=in->inputTensors[1].geometry;const auto&x=in->inputTensors[2].geometry;const auto&lut=in->inputTensors[3].geometry;const auto&ids=in->inputTensors[4].geometry;const auto&y=in->outputTensors[0].geometry;
 if(w.dataType!=DATA_U8||s.dataType!=DATA_U8||x.dataType!=DATA_BF16||lut.dataType!=DATA_BF16||ids.dataType!=DATA_I32||y.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_DATA_TYPE;
 const uint64_t k=ids.maxSizes[0],t=ids.maxSizes[1],slots=k*t,eblocks=w.maxSizes[1]/256;
 if(!k||k>384||!t||slots>uint64_t(INT_MAX)/6144||eblocks<12||eblocks%12||!geometry(ids,DATA_I32,k,t)||!geometry(w,DATA_U8,256,eblocks*256)||!geometry(s,DATA_U8,512,eblocks*8)||!geometry(x,DATA_BF16,256,slots)||!geometry(lut,DATA_BF16,512,1)||!geometry(y,DATA_F32,slots*6144,1))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<5;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&a=out->inputTensorAccessPattern[2];a.allRequired=0;a.mapping[0]={0,0,0,255};a.mapping[1]={0,1.f/12,-1,1};
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=slots*12;auto&p=out->outputTensorAccessPattern[0];p.mapping[0]={0,512,0,511};p.mapping[1]={0,0,0,0};out->kernel.paramsNr=1;out->kernel.scalarParams[0]=k==1?0:uint32_t(((uint64_t(1)<<32)+(k-1))/k);
 unsigned char*start=(id==1)?&_binary_control_o_start:&_binary_candidate_o_start;unsigned char*end=(id==1)?&_binary_control_o_end:&_binary_candidate_o_end;unsigned size=end-start,cap=out->kernel.elfSize;out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
