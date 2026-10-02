// Requires the runtime-matching modern TPC SDK. Never compile this against old gcapi.
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdint>
#include <initializer_list>
using namespace tpc_lib_api;
#define ELFS(X) X(decode_native) X(decode_rows) X(decode_rows256)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_mx4c3_native_s2_252_bf16","gk_mx4c3_rows_s2_252_bf16","gk_mx4c3_rows256_s2_252_bf16"};
static int which(const char*n){for(int i=0;i<3;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T> static void map(T&p,int dim,int index,float stride,int lo,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?3:0;if(g&&cap<*n)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;if(g)for(unsigned i=0;i<*n;++i){g[i]={};std::strcpy(g[i].name,names[i]);}return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=3||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->deviceId!=DEVICE_ID_GAUDI2)return GLUE_NODE_NOT_FOUND;
 const auto&y=in->outputTensors[0].geometry;
 const auto&wgeo=in->inputTensors[0].geometry;const auto&sgeo=in->inputTensors[1].geometry;
 const auto&lut=in->inputTensors[2].geometry;
 if(y.dataType!=DATA_BF16||wgeo.dataType!=DATA_U8||sgeo.dataType!=DATA_U8||lut.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
 if(y.dims!=2||wgeo.dims!=unsigned(id?2:3)||sgeo.dims!=wgeo.dims||lut.dims!=2||lut.maxSizes[0]!=512||lut.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(const auto*geometry:{&y,&wgeo,&sgeo,&lut}){
  uint64_t count=1;for(unsigned d=0;d<geometry->dims;++d){
   if(!geometry->maxSizes[d]||geometry->maxSizes[d]>INT32_MAX)return GLUE_INCOMPATIBLE_INPUT_SIZE;
   if(geometry->minSizes[d]!=geometry->maxSizes[d])return GLUE_UNSUPPORTED_DYNAMIC_SHAPE;
   if(count>uint64_t(INT32_MAX)/geometry->maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
   count*=geometry->maxSizes[d];
  }
 }
 std::memset(out->inputTensorAccessPattern,0,3*sizeof(TensorAccessPattern));
 std::memset(out->outputTensorAccessPattern,0,sizeof(TensorAccessPattern));
 out->auxiliaryTensorNr=0;
 out->indexSpaceRank=2;out->indexSpaceGeometry[1]=1;out->kernel.paramsNr=0;
 for(unsigned i=0;i<3;++i)out->inputTensorAccessPattern[i].allRequired=0;
 out->inputTensorAccessPattern[2].allRequired=1;out->outputTensorAccessPattern[0].allRequired=0;
 auto&w=out->inputTensorAccessPattern[0];auto&s=out->inputTensorAccessPattern[1];auto&o=out->outputTensorAccessPattern[0];
 if(id==0){
  if(in->nodeParams.nodeParamsSize!=8||!in->nodeParams.nodeParams)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const int*p=static_cast<const int*>(in->nodeParams.nodeParams);int block=p[0],pair=p[1];
  int n=y.maxSizes[0],k=y.maxSizes[1];
  if((n!=256&&n!=512)||k<32||k%32||block<0||block>=int(wgeo.maxSizes[2])||pair<0||pair>1||pair+n/256>2||wgeo.maxSizes[0]!=256||wgeo.maxSizes[1]!=unsigned(k)||sgeo.maxSizes[0]!=512||sgeo.maxSizes[1]!=unsigned(k/32)||sgeo.maxSizes[2]!=wgeo.maxSizes[2])return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=2;out->kernel.scalarParams[0]=block;out->kernel.scalarParams[1]=pair;
  out->indexSpaceGeometry[0]=k/32;
  map(w,0,0,0,pair*128,pair*128+n/2-1);map(w,1,0,32,0,31);map(w,2,0,0,block,block);
  map(s,0,0,0,pair*256,pair*256+n-1);map(s,1,0,1,0,0);map(s,2,0,0,block,block);
  map(o,0,0,0,0,n-1);map(o,1,0,32,0,31);
 }else{
  int k=y.maxSizes[0],n=y.maxSizes[1];
  if(in->nodeParams.nodeParamsSize||n<1||k<1||(k>=32&&k%32)||wgeo.maxSizes[0]!=unsigned((k+1)/2)||wgeo.maxSizes[1]!=unsigned(n)||sgeo.maxSizes[0]!=unsigned((k+31)/32)||sgeo.maxSizes[1]!=unsigned(n))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  int task=id==2?256:32,count=k<task?k:task,bytes=(count+1)/2,groups=(count+31)/32;
  out->indexSpaceGeometry[0]=(uint64_t(k)+task-1)/task;out->indexSpaceGeometry[1]=n;
  map(w,0,0,task/2,0,bytes-1);map(w,1,1,1,0,0);
  map(s,0,0,task/32,0,groups-1);map(s,1,1,1,0,0);
  map(o,0,0,task,0,count-1);map(o,1,1,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};
 unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 3;}
