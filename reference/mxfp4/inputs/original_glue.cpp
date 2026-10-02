#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELF(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELF(gemv) ELF(prep) ELF(gate_prepare) ELF(combine) ELF(rank_sum)
static const char* names[]={"nm_gemv","nm_prep","nm_gate_prepare","nm_combine","nm_rank_sum"};
static int which(const char*n){for(int i=0;i<5;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId dev,uint32_t*n,GuidInfo*g){*n=dev==DEVICE_ID_GAUDI2?5:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 const unsigned ni[]={5,2,2,3,1},no[]={1,2,2,1,1};
 if(in->inputTensorNr!=ni[id]||in->outputTensorNr!=no[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<ni[id];++i)out->inputTensorAccessPattern[i].allRequired=1;
 out->indexSpaceRank=1;
 auto linear=[&](unsigned output,int stride,int length,int dim=0){
  auto&p=out->outputTensorAccessPattern[output];
  p.mapping[0].indexSpaceDim=0;p.mapping[0].a=dim==0?stride:0;p.mapping[0].start_b=0;p.mapping[0].end_b=length-1;
  p.mapping[1].indexSpaceDim=0;p.mapping[1].a=dim==1?stride:0;p.mapping[1].start_b=p.mapping[1].end_b=0;
 };
 auto& y=in->outputTensors[0].geometry;
 if(id==0){
  int kp=in->inputTensors[2].geometry.maxSizes[0],tasks=in->inputTensors[2].geometry.maxSizes[1];
  if(kp%32||y.maxSizes[0]!=tasks*512||in->inputTensors[4].geometry.maxSizes[0]!=tasks)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=tasks;linear(0,512,512);
  auto&a=out->inputTensorAccessPattern[2];a.allRequired=0;
  a.mapping[0].indexSpaceDim=0;a.mapping[0].a=0;a.mapping[0].start_b=0;a.mapping[0].end_b=kp-1;
  a.mapping[1].indexSpaceDim=0;a.mapping[1].a=1;a.mapping[1].start_b=a.mapping[1].end_b=0;
 }else if(id==1){
  if(y.maxSizes[0]%128||in->nodeParams.nodeParamsSize!=12)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,12);
  out->indexSpaceGeometry[0]=y.maxSizes[1];linear(0,1,y.maxSizes[0],1);linear(1,1,1);
 }else if(id==2){
  out->indexSpaceGeometry[0]=y.maxSizes[1];linear(0,1,y.maxSizes[0],1);linear(1,1,1);
  out->kernel.paramsNr=1;out->kernel.scalarParams[0]=in->inputTensors[0].geometry.maxSizes[0]/(2*y.maxSizes[0]*in->inputTensors[1].geometry.maxSizes[0]*in->inputTensors[1].geometry.maxSizes[1]);
 }else if(id==4){
  if(in->nodeParams.nodeParamsSize!=4||y.maxSizes[0]%128)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  out->indexSpaceGeometry[0]=y.maxSizes[0]*y.maxSizes[1]/128;
  out->outputTensorAccessPattern[0].allRequired=1;
 }else{
  if(in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  out->indexSpaceGeometry[0]=y.maxSizes[0]*y.maxSizes[1]/128;
  out->outputTensorAccessPattern[0].allRequired=1;
 }
 unsigned char* start[]={&_binary_gemv_o_start,&_binary_prep_o_start,&_binary_gate_prepare_o_start,&_binary_combine_o_start,&_binary_rank_sum_o_start};
 unsigned char* end[]={&_binary_gemv_o_end,&_binary_prep_o_end,&_binary_gate_prepare_o_end,&_binary_combine_o_end,&_binary_rank_sum_o_end};
 unsigned cap=out->kernel.elfSize,size=end[id]-start[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){
 if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
