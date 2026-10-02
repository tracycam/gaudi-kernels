#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(group_prepare) X(group_decode) X(group_combine) X(group_gate) X(group_legacy_gate) X(group_decode_legacy)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"gk_mxfp4_group_prepare","gk_mxfp4_group_decode","gk_mxfp4_group_combine","gk_mxfp4_group_gate","gk_mxfp4_group_legacy_gate","gk_mxfp4_group_decode_legacy_n512"};
static int which(const char*n){for(int i=0;i<6;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?6:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;unsigned count=id==0?2:(id==1||id==5)?4:id==4?2:5,outputs=id==0?4:1,bytes=id==0?12:(id==1||id==5)?16:id==4?4:8;
 if(in->inputTensorNr!=count||in->outputTensorNr!=outputs||in->nodeParams.nodeParamsSize!=bytes)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 out->kernel.paramsNr=bytes/4;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,bytes);
 for(unsigned i=0;i<count;++i)out->inputTensorAccessPattern[i].allRequired=1;const auto&y=in->outputTensors[0].geometry;auto&o=out->outputTensorAccessPattern[0];
 if(id==0){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[2];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,0,0,y.maxSizes[1]-1);map(o,2,0,1,0,0);out->outputTensorAccessPattern[1].allRequired=1;for(unsigned i=2;i<4;++i){auto&p=out->outputTensorAccessPattern[i];map(p,0,0,1,0,0);map(p,1,0,0,0,0);}}
 else if(id==1||id==5){if(y.maxSizes[0]%512||y.maxSizes[1]%32)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;out->indexSpaceRank=3;out->indexSpaceGeometry[0]=y.maxSizes[1]/32;out->indexSpaceGeometry[1]=y.maxSizes[0]/512;out->indexSpaceGeometry[2]=y.maxSizes[2];map(o,0,1,512,0,511);map(o,1,0,32,0,31);map(o,2,2,1,0,0);}
 else if(id==4){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[1];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,1,0,0);}
 else {unsigned tile=id==3?128:64;out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+tile-1)/tile;out->indexSpaceGeometry[1]=y.maxSizes[1];map(o,0,0,tile,0,tile-1);map(o,1,1,1,0,0);}
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
