#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(gemv1) X(gemv2) X(reduce_bf16) X(reduce_f32) X(reduce_bias_bf16) X(reduce_bias_f32) X(bias_bf16) X(bias_f32) X(gemv_acc321) X(gemv_acc322) X(rowdot_bf16) X(rowdot_f32) X(rowdot_bias_bf16) X(rowdot_bias_f32) X(rowdot4_bf16) X(rowdot4_f32) X(rowdot4_bias_bf16) X(rowdot4_bias_f32)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_bf16_gemv1","gk_bf16_gemv2","gk_bf16_reduce_bf16","gk_bf16_reduce_f32","gk_bf16_reduce_bias_bf16","gk_bf16_reduce_bias_f32","gk_bf16_bias_bf16","gk_bf16_bias_f32","gk_bf16_gemv_acc321","gk_bf16_gemv_acc322","gk_bf16_rowdot_bf16","gk_bf16_rowdot_f32","gk_bf16_rowdot_bias_bf16","gk_bf16_rowdot_bias_f32","gk_bf16_rowdot4_bf16","gk_bf16_rowdot4_f32","gk_bf16_rowdot4_bias_bf16","gk_bf16_rowdot4_bias_f32"};
static int which(const char*n){for(int i=0;i<18;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int dim,int index,float stride,int lo,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?18:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=(id>=10&&((id-10)%4)>=2?3:id>=10?2:((id<2||id>=4)?2:1))||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 out->indexSpaceRank=2;auto&y=in->outputTensors[0].geometry;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=0;
 out->outputTensorAccessPattern[0].allRequired=0;
 out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;
 if(id>=10){
  out->indexSpaceGeometry[0]=y.maxSizes[0];out->indexSpaceGeometry[1]=y.maxSizes[1];
  int k=in->inputTensors[0].geometry.maxSizes[0];
  map(out->inputTensorAccessPattern[0],0,0,0,0,k-1);map(out->inputTensorAccessPattern[0],1,1,1,0,0);
  map(out->inputTensorAccessPattern[1],0,0,0,0,k-1);map(out->inputTensorAccessPattern[1],1,0,1,0,0);
  if(((id-10)%4)>=2){map(out->inputTensorAccessPattern[2],0,0,1,0,0);map(out->inputTensorAccessPattern[2],1,1,0,0,0);}
  map(out->outputTensorAccessPattern[0],0,0,1,0,0);map(out->outputTensorAccessPattern[0],1,1,1,0,0);
 }else if(id<2||id>=8){
  out->indexSpaceGeometry[1]=y.maxSizes[2];int rows=id>=8?id-7:id+1;
  map(out->inputTensorAccessPattern[0],0,1,256,0,255);map(out->inputTensorAccessPattern[0],1,0,0,0,rows-1);
  map(out->inputTensorAccessPattern[1],0,0,128,0,127);map(out->inputTensorAccessPattern[1],1,1,256,0,255);
  map(out->outputTensorAccessPattern[0],0,0,128,0,127);map(out->outputTensorAccessPattern[0],1,0,0,0,rows-1);map(out->outputTensorAccessPattern[0],2,1,1,0,0);
 }else{
  out->indexSpaceGeometry[1]=y.maxSizes[1];
  map(out->inputTensorAccessPattern[0],0,0,128,0,127);map(out->inputTensorAccessPattern[0],1,1,1,0,0);
  if(id<6)map(out->inputTensorAccessPattern[0],2,0,0,0,in->inputTensors[0].geometry.maxSizes[2]-1);
  if(id>=4){map(out->inputTensorAccessPattern[1],0,0,128,0,127);map(out->inputTensorAccessPattern[1],1,1,0,0,0);}
  map(out->outputTensorAccessPattern[0],0,0,128,0,127);map(out->outputTensorAccessPattern[0],1,1,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};
 unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.paramsNr=0;out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
