#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(dequant_bf16) X(activation_native) X(reduce1) X(reduce4) X(dequant_native_f32scale) X(dequant_native_bf16scale) X(reduce1bf16) X(reduce4bf16)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
#undef DECL
static const char* names[]={"mac_original_scaled_bf16","mac_activation_native","mac_block_reduce1","mac_block_reduce4","mac_native_scaled_bf16","mac_native_scaled_bf16_fast","mac_block_reduce1_bf16","mac_block_reduce4_bf16"};
static int which(const char*n){for(int i=0;i<8;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T> static void map(T&p,int dim,int index,float stride,int lo,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?8:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=((id==0||id==4||id==5)?2:id==1?1:3)||in->outputTensorNr!=(id==1?2:1))return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=0;
 for(unsigned i=0;i<in->outputTensorNr;++i)out->outputTensorAccessPattern[i].allRequired=0;
 out->indexSpaceRank=2;
 const auto& x=in->inputTensors[0].geometry;const auto& y=in->outputTensors[0].geometry;
 if(id==0||id==4||id==5){
  out->indexSpaceGeometry[0]=x.maxSizes[2];out->indexSpaceGeometry[1]=(x.maxSizes[1]+127)/128;
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,0,0,127);map(a,1,1,128,0,127);map(a,2,0,1,0,0);
  auto&s=out->inputTensorAccessPattern[1];map(s,0,0,1,0,0);map(s,1,1,1,0,0);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,128,0,127);
 }else if(id==1){
  out->indexSpaceGeometry[0]=x.maxSizes[0]/128;out->indexSpaceGeometry[1]=x.maxSizes[1];
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,128,0,127);map(a,1,1,1,0,0);
  for(int i=0;i<2;++i){auto&o=out->outputTensorAccessPattern[i];map(o,0,0,0,0,i==0?127:0);map(o,1,1,1,0,0);map(o,2,0,1,0,0);}
 }else{
  unsigned rows=(id==2||id==6)?1:4,g=x.maxSizes[2];out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=(y.maxSizes[1]+rows-1)/rows;
  auto&p=out->inputTensorAccessPattern[0];map(p,0,0,128,0,127);map(p,1,1,rows,0,rows-1);map(p,2,0,0,0,g-1);
  auto&a=out->inputTensorAccessPattern[1];map(a,0,0,0,0,0);map(a,1,1,rows,0,rows-1);map(a,2,0,0,0,g-1);
  auto&w=out->inputTensorAccessPattern[2];map(w,0,0,0,0,g-1);map(w,1,0,1,0,0);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,rows,0,rows-1);
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
