#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
#define ELFS(X) X(block_quant) X(block_decode) X(block_decode_fast) X(block_reduce) X(block_finish)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_block128_quant_v1","gk_block128_decode_v1","gk_block128_decode_fast_v1","gk_block128_reduce_bias_v1","gk_block128_finish_v1"};
static int which(const char*n){for(int i=0;i<5;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,float stride,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=stride;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?5:0;if(g)for(unsigned i=0;i<cap&&i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 unsigned inputs=id==0?1:id==3?4:2,outputs=id==0?2:1;
 if(in->inputTensorNr!=inputs||in->outputTensorNr!=outputs)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<inputs;++i){const auto&g=in->inputTensors[i].geometry;for(unsigned d=0;d<g.dims;++d)if(!g.maxSizes[d]||g.maxSizes[d]>INT32_MAX-128)return GLUE_INCOMPATIBLE_INPUT_SIZE;out->inputTensorAccessPattern[i].allRequired=0;}
 for(unsigned i=0;i<outputs;++i)out->outputTensorAccessPattern[i].allRequired=0;
 const auto&x=in->inputTensors[0].geometry;const auto&y=in->outputTensors[0].geometry;
 out->indexSpaceRank=2;
 if(id==0){
  out->indexSpaceGeometry[0]=(x.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=x.maxSizes[1];
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,128,0,127);map(a,1,1,1,0,0);
  for(unsigned i=0;i<2;++i){auto&o=out->outputTensorAccessPattern[i];map(o,0,0,0,0,i==0?127:0);map(o,1,1,1,0,0);map(o,2,0,1,0,0);}
 }else if(id==1||id==2){
  out->indexSpaceGeometry[0]=x.maxSizes[2];out->indexSpaceGeometry[1]=(x.maxSizes[1]+127)/128;
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,0,0,127);map(a,1,1,128,0,127);map(a,2,0,1,0,0);
  auto&s=out->inputTensorAccessPattern[1];map(s,0,0,1,0,0);map(s,1,1,1,0,0);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,128,0,127);
 }else if(id==3){
  unsigned rows=4,g=x.maxSizes[2];out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=(y.maxSizes[1]+rows-1)/rows;
  auto&p=out->inputTensorAccessPattern[0];map(p,0,0,128,0,127);map(p,1,1,rows,0,rows-1);map(p,2,0,0,0,g-1);
  auto&a=out->inputTensorAccessPattern[1];map(a,0,0,0,0,0);map(a,1,1,rows,0,rows-1);map(a,2,0,0,0,g-1);
  auto&w=out->inputTensorAccessPattern[2];map(w,0,0,0,0,g-1);map(w,1,0,1,0,0);
  auto&b=out->inputTensorAccessPattern[3];map(b,0,0,128,0,127);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,rows,0,rows-1);
 }else{
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=y.maxSizes[1];
  auto&p=out->inputTensorAccessPattern[0];map(p,0,0,128,0,127);map(p,1,1,1,0,0);
  auto&b=out->inputTensorAccessPattern[1];map(b,0,0,128,0,127);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,1,0,0);
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
