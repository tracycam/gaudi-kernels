#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(activation) X(decode) X(epilogue8) X(epilogue16) X(decode128) X(activation_fast) X(epilogue8_fast) X(epilogue16_fast) X(epilogue8_rows) X(epilogue16_rows)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
#undef DECL
static const char* names[]={"fp8_linear_activation","fp8_linear_decode","fp8_linear_epilogue8","fp8_linear_epilogue16","fp8_linear_decode128","fp8_linear_activation_fast","fp8_linear_epilogue8_fast","fp8_linear_epilogue16_fast","fp8_linear_epilogue8_rows","fp8_linear_epilogue16_rows"};
static int which(const char*n){for(int i=0;i<10;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T> static void map(T&p,int dim,int index,float stride,int lo,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?10:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 const unsigned inputs[]={1,1,4,3,1,1,4,3,4,3},outputs[]={2,1,1,1,1,2,1,1,1,1};
 if(in->inputTensorNr!=inputs[id]||in->outputTensorNr!=outputs[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=0;
 for(unsigned i=0;i<in->outputTensorNr;++i)out->outputTensorAccessPattern[i].allRequired=0;
 const auto& x=in->inputTensors[0].geometry;const auto& y=in->outputTensors[0].geometry;
 out->indexSpaceRank=2;
 if(id==0||id==5){
  out->indexSpaceGeometry[0]=x.maxSizes[1];out->indexSpaceGeometry[1]=1;
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,0,0,x.maxSizes[0]-1);map(a,1,0,1,0,0);
  for(int i=0;i<2;++i){auto&o=out->outputTensorAccessPattern[i];map(o,0,0,0,0,i==0?x.maxSizes[0]-1:0);map(o,1,0,1,0,0);}
 }else if(id==1||id==4){
  unsigned rb=id==1?8:128;
  out->indexSpaceGeometry[0]=(x.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=(x.maxSizes[1]+rb-1)/rb;
  auto&a=out->inputTensorAccessPattern[0];map(a,0,0,128,0,127);map(a,1,1,rb,0,rb-1);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,rb,0,rb-1);
 }else{
  unsigned tile=(id==6||id==7)?512:128,rows=(id==8||id==9)?4:1;
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+tile-1)/tile;out->indexSpaceGeometry[1]=(y.maxSizes[1]+rows-1)/rows;
  auto&p=out->inputTensorAccessPattern[0];map(p,0,0,tile,0,tile-1);map(p,1,1,rows,0,rows-1);
  if(id==2||id==6||id==8){auto&a=out->inputTensorAccessPattern[1];map(a,0,0,0,0,0);map(a,1,1,rows,0,rows-1);}
  for(unsigned i=(id==2||id==6||id==8)?2:1;i<inputs[id];++i){auto&w=out->inputTensorAccessPattern[i];map(w,0,0,tile,0,tile-1);map(w,1,0,0,0,0);}
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,tile,0,tile-1);map(o,1,1,rows,0,rows-1);
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
