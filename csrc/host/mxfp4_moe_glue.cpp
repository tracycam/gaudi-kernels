#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(moe_direct) X(moe_fused) X(moe_combine) X(moe_fused256) X(moe_fused256c) X(moe_legacy_down) X(moe_legacy_combine)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"gk_mxfp4_moe_direct512","gk_mxfp4_moe_fused512","gk_mxfp4_moe_combine_f32","gk_mxfp4_moe_fused256","gk_mxfp4_moe_fused256c","gk_mxfp4_moe_legacy_down","gk_mxfp4_moe_legacy_combine"};
static int which(const char*n){for(int i=0;i<7;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?7:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;unsigned count=id==5?5:id==6?3:id!=2?6:3;
 if(in->inputTensorNr!=count||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto&y=in->outputTensors[0].geometry;unsigned width=y.maxSizes[0],rows=y.maxSizes[1];
 if(!width||width%512||!rows)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 unsigned bytes=id==5?0:id==6?4:id!=2?16:4;if(in->nodeParams.nodeParamsSize!=bytes)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 out->kernel.paramsNr=bytes/4;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,bytes);
 for(unsigned i=0;i<count;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&o=out->outputTensorAccessPattern[0];
 if(id==5){out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/in->inputTensors[4].geometry.maxSizes[0]);out->indexSpaceRank=1;out->indexSpaceGeometry[0]=width/512;map(o,0,0,512,0,511);map(o,1,0,0,0,0);}
 else if(id==6){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=width*rows/128;o.allRequired=1;}
 else if(id!=2){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=width/(id>=3?256:512)*rows;o.allRequired=1;}
 else {out->indexSpaceRank=2;out->indexSpaceGeometry[0]=width/64;out->indexSpaceGeometry[1]=rows;map(o,0,0,64,0,63);map(o,1,1,1,0,0);}
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
