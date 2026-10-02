#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
#define ELFS(X) X(down_p4) X(down_p6) X(down_old)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* guids[]={"gk_down_combine_m1_p4_v0","gk_down_combine_m1_p6_v0","gk_down_combine_old_body_v0"};
static int which(const char* name){for(int i=0;i<3;++i)if(!std::strcmp(name,guids[i]))return i;return -1;}
static bool geometry(const Tensor& t,TensorDataType dtype,unsigned w,unsigned h){return t.geometry.dataType==dtype&&t.geometry.dims==2&&t.geometry.maxSizes[0]==w&&t.geometry.maxSizes[1]==h;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t*count,GuidInfo*names){unsigned cap=*count;*count=device==DEVICE_ID_GAUDI2?3:0;if(names)for(unsigned i=0;i<cap&&i<*count;++i)std::strcpy(names[i].name,guids[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;bool old=id==2;unsigned count=old?5:7;
 if(in->inputTensorNr!=count||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto*w=in->inputTensors;unsigned width=old?256:128,rows_per_expert=old?3072:6144;
 auto rows=w[0].geometry.maxSizes[1];if(rows%rows_per_expert||rows/rows_per_expert<8||rows/rows_per_expert>384)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!geometry(w[0],DATA_U8,width,rows)||!geometry(w[1],DATA_U8,width*2,rows/32)||!geometry(w[2],DATA_BF16,256,8)||!geometry(w[3],DATA_BF16,512,1)||!geometry(w[4],DATA_I32,8,1))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!old&&(!geometry(w[5],DATA_F32,8,1)||!geometry(w[6],DATA_U8,256,2)))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!geometry(in->outputTensors[0],DATA_F32,old?49152:6144,1))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 if(in->nodeParams.nodeParamsSize)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=old?96:24;
 for(unsigned i=0;i<count;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto& access=out->outputTensorAccessPattern[0];access.mapping[0]={0,old?512.f:256.f,0,old?511:255};access.mapping[1]={0,0,0,0};
 out->kernel.paramsNr=old?1:0;if(old)out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/8);
#define START(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(START)},*end[]={ELFS(END)};unsigned size=end[id]-begin[id],cap=out->kernel.elfSize;out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(ls&&cap){for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
