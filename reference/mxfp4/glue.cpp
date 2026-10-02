#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELF(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELF(direct_gp) ELF(compact_gate) ELF(direct_down) ELF(gate_broadcast) ELF(sorted_prep) ELF(sorted_combine) ELF(masked_gemv)
const char* names[]={"ub_direct_gp","ub_compact_gate","ub_direct_down","ub_gate_broadcast","ub_sorted_prep","ub_sorted_combine","ub_masked_gemv"};
int which(const char*n){for(int i=0;i<7;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?7:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 unsigned ni[]={5,2,5,2,3,4,5},no[]={1,1,1,2,3,1,1};
 if(in->inputTensorNr!=ni[id]||in->outputTensorNr!=no[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<ni[id];++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto& y=in->outputTensors[0].geometry;
 out->indexSpaceRank=1;
 auto&p=out->outputTensorAccessPattern[0];
 if(id==6){
  if(in->inputTensors[4].geometry.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_DATA_TYPE;
  const auto& x=in->inputTensors[2].geometry;auto tasks=x.maxSizes[1];
  if(x.maxSizes[0]%32||y.maxSizes[0]!=tasks*512)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=tasks;p.mapping[0]={0,512,0,511};p.mapping[1]={0,0,0,0};
  auto&a=out->inputTensorAccessPattern[2];a.allRequired=0;
  a.mapping[0]={0,0,0,float(x.maxSizes[0]-1)};a.mapping[1]={0,1,0,0};
 }else if(id==3){
  if(y.maxSizes[0]!=256||in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  out->indexSpaceGeometry[0]=in->inputTensors[1].geometry.maxSizes[0]*in->inputTensors[1].geometry.maxSizes[1];
  p.mapping[0]={0,0,0,255};p.mapping[1]={0,12,0,11};
  auto&q=out->outputTensorAccessPattern[1];q.mapping[0]={0,12,0,11};q.mapping[1]={0,0,0,0};
 }else if(id==4){
  if(in->inputTensors[1].geometry.dataType!=DATA_I32||in->inputTensors[2].geometry.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(in->nodeParams.nodeParamsSize!=12)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,12);
  out->indexSpaceGeometry[0]=y.maxSizes[1];
  p.mapping[0]={0,0,0,float(y.maxSizes[0]-1)};p.mapping[1]={0,1,0,0};
  auto&q=out->outputTensorAccessPattern[1];q.mapping[0]={0,1,0,0};q.mapping[1]={0,0,0,0};
  out->outputTensorAccessPattern[2].allRequired=1;
 }else if(id==5){
  if(in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  out->indexSpaceGeometry[0]=y.maxSizes[0]*y.maxSizes[1]/128;p.allRequired=1;
 }else if(id==1){
  if(y.maxSizes[0]!=256||in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  out->indexSpaceGeometry[0]=y.maxSizes[1];
  auto&input=out->inputTensorAccessPattern[0];input.allRequired=0;
  input.mapping[0].indexSpaceDim=0;input.mapping[0].a=1536;input.mapping[0].start_b=0;input.mapping[0].end_b=1535;
  input.mapping[1].indexSpaceDim=0;input.mapping[1].a=0;input.mapping[1].start_b=input.mapping[1].end_b=0;
  p.mapping[0].indexSpaceDim=0;p.mapping[0].a=0;p.mapping[0].start_b=0;p.mapping[0].end_b=255;
  p.mapping[1].indexSpaceDim=0;p.mapping[1].a=1;p.mapping[1].start_b=p.mapping[1].end_b=0;
 }else{
  if(in->inputTensors[2].geometry.dataType!=DATA_BF16||in->inputTensors[4].geometry.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(in->inputTensors[2].geometry.maxSizes[0]!=(id==0?6144:256)||y.maxSizes[0]%512)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  auto topk=in->inputTensors[4].geometry.maxSizes[0];
  if(topk<1||topk>384)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/topk);
  // Conservative affine bounds include floor(task / stride), including FP rounding.
  // Do not declare every task consumes the entire activation tensor: that prevents
  // the compiler from slicing the compact producer/consumer dataflow.
  auto&input=out->inputTensorAccessPattern[2];input.allRequired=0;
  input.mapping[0].indexSpaceDim=0;input.mapping[0].a=0;input.mapping[0].start_b=0;input.mapping[0].end_b=id==0?6143:255;
  input.mapping[1].indexSpaceDim=0;input.mapping[1].a=1.f/(id==0?3*topk:12);input.mapping[1].start_b=-1;input.mapping[1].end_b=1;
  out->indexSpaceGeometry[0]=y.maxSizes[0]/512;
  p.mapping[0].indexSpaceDim=0;p.mapping[0].a=512;p.mapping[0].start_b=0;p.mapping[0].end_b=511;
  p.mapping[1].indexSpaceDim=0;p.mapping[1].a=0;p.mapping[1].start_b=p.mapping[1].end_b=0;
 }
 unsigned char* starts[]={&_binary_direct_gp_o_start,&_binary_compact_gate_o_start,&_binary_direct_down_o_start,&_binary_gate_broadcast_o_start,&_binary_sorted_prep_o_start,&_binary_sorted_combine_o_start,&_binary_masked_gemv_o_start};
 unsigned char* ends[]={&_binary_direct_gp_o_end,&_binary_compact_gate_o_end,&_binary_direct_down_o_end,&_binary_gate_broadcast_o_end,&_binary_sorted_prep_o_end,&_binary_sorted_combine_o_end,&_binary_masked_gemv_o_end};
 unsigned cap=out->kernel.elfSize,size=ends[id]-starts[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,starts[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){
 if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
