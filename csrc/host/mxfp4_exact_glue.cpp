#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(mx_exact_v2) X(mx_exact_v2_bias) X(mx_repair_v2) X(mx_repair_v2_bias) X(mx_exact_v3) X(mx_exact_v3_bias) X(mx_repair_v3) X(mx_repair_v3_bias) X(mx_dispatch_prepare) X(mx_dispatch_prepare_bias) X(mx_dispatch_history) X(mx_dispatch_finish) X(mx_dispatch_finish_bias)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"gk_mxfp4_integer_exact_v2","gk_mxfp4_integer_exact_v2_bias","gk_mxfp4_integer_repair_v2","gk_mxfp4_integer_repair_v2_bias","gk_mxfp4_integer_exact_v3","gk_mxfp4_integer_exact_v3_bias","gk_mxfp4_integer_repair_v3","gk_mxfp4_integer_repair_v3_bias","gk_mxfp4_dispatch_prepare","gk_mxfp4_dispatch_prepare_bias","gk_mxfp4_dispatch_history","gk_mxfp4_dispatch_finish","gk_mxfp4_dispatch_finish_bias"};
static int which(const char*name){for(int i=0;i<13;++i)if(!std::strcmp(name,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?13:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 unsigned count=id<8?3+unsigned(bool(id&2))+unsigned(bool(id&1)):id==8?1:id==9?2:id==10?6:id==11?5:6;
 unsigned outputs=id==8||id==9?2:1;
 if(in->inputTensorNr!=count||in->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto&y=in->outputTensors[0].geometry;
 for(unsigned i=0;i<count;++i)out->inputTensorAccessPattern[i].allRequired=1;
 if(id==8||id==9){
  if(in->nodeParams.nodeParamsSize!=28)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const int*params=(const int*)in->nodeParams.nodeParams;int splits=params[3];
  if(splits<1||y.maxSizes[1]%splits)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=7;std::memcpy(out->kernel.scalarParams,params,28);
  out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[1]/splits;
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,splits,0,splits-1);
  auto&f=out->outputTensorAccessPattern[1];map(f,0,0,0,0,0);map(f,1,0,splits,0,splits-1);
 }else if(id==10){
  unsigned kp=in->inputTensors[2].geometry.maxSizes[0],tasks=in->inputTensors[2].geometry.maxSizes[1];
  if(!kp||kp%32||y.maxSizes[0]!=tasks*512)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=0;out->indexSpaceRank=1;out->indexSpaceGeometry[0]=tasks;
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,512,0,511);map(o,1,0,0,0,0);
  auto&x=out->inputTensorAccessPattern[2];x.allRequired=0;map(x,0,0,0,0,kp-1);map(x,1,0,1,0,0);
 }else{
  unsigned params_bytes=id<8?8:12;
  if(in->nodeParams.nodeParamsSize!=params_bytes)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const int*params=(const int*)in->nodeParams.nodeParams;
  if(params[0]<1||params[1]<1||y.maxSizes[0]!=unsigned(params[0])||in->inputTensors[2].geometry.maxSizes[0]!=unsigned(params[1])||in->inputTensors[2].geometry.maxSizes[1]!=y.maxSizes[1])return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=params_bytes/4;std::memcpy(out->kernel.scalarParams,params,params_bytes);
  unsigned tile=id>=11?256:512;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+tile-1)/tile;out->indexSpaceGeometry[1]=y.maxSizes[1];
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,tile,0,tile-1);map(o,1,1,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
