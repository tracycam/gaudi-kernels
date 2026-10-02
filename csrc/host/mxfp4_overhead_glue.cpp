#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(oh_history) X(oh_guarded) X(oh_exact) X(oh_prepare) X(oh_finish) X(oh_finish_bias) X(oh_prologue)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"gk_mxfp4_history","gk_mxfp4_guarded","gk_mxfp4_exact512","gk_mxfp4_prepare","gk_mxfp4_finish","gk_mxfp4_finish_bias","gk_mxfp4_prologue"};
static int which(const char*n){for(int i=0;i<7;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?7:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 unsigned inputs=id==0||id==6?5:id==1||id==2?6:id==5?2:1,outputs=id==3?2:1;
 if(in->inputTensorNr!=inputs||in->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 out->kernel.paramsNr=0;for(unsigned i=0;i<inputs;++i)out->inputTensorAccessPattern[i].allRequired=1;
 const auto&y=in->outputTensors[0].geometry;out->indexSpaceRank=1;
 if(id==0||id==1||id==2||id==6){
  unsigned kp=in->inputTensors[2].geometry.maxSizes[0],tasks=in->inputTensors[2].geometry.maxSizes[1];
  if(kp%32||y.maxSizes[0]!=tasks*512||in->inputTensors[4].geometry.maxSizes[0]!=tasks)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=tasks;auto&o=out->outputTensorAccessPattern[0];map(o,0,0,512,0,511);map(o,1,0,0,0,0);
  auto&x=out->inputTensorAccessPattern[2];x.allRequired=0;map(x,0,0,0,0,kp-1);map(x,1,0,1,0,0);
 }else if(id==3){
  if(in->nodeParams.nodeParamsSize!=8)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,8);
  out->indexSpaceGeometry[0]=y.maxSizes[1];
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,1,0,0);
  auto&f=out->outputTensorAccessPattern[1];map(f,0,0,0,0,0);map(f,1,0,1,0,0);
 }else{
  if(in->nodeParams.nodeParamsSize!=8)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,8);
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+63)/64;out->indexSpaceGeometry[1]=y.maxSizes[1];
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,64,0,63);map(o,1,1,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
