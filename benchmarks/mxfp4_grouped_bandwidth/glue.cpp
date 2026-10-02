#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#ifndef GK_GROUPED_TILE
#define GK_GROUPED_TILE 256
#endif
#define ELFS(X) X(grouped) X(decode) X(reduce) X(ring_gate)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_grouped_mxfp4_smallm","gk_grouped_mxfp4_decode","gk_grouped_mxfp4_reduce","gk_grouped_ring_gate"};
static int which(const char*n){for(int i=0;i<4;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int dim,int idx,float a,int lo,int hi){p.mapping[dim].indexSpaceDim=idx;p.mapping[dim].a=a;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?4:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 const int small_inputs=
#ifdef GK_LITERAL_GP
 5;
#else
 6;
#endif
 if(in->inputTensorNr!=(id==0?small_inputs:(id==1||id==3)?4:1)||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto& y=in->outputTensors[0].geometry;out->kernel.paramsNr=0;out->indexSpaceRank=3;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=0;
 auto&o=out->outputTensorAccessPattern[0];o.allRequired=0;
 if(id==0){
#ifdef GK_LITERAL_GP
  int e=in->inputTensors[4].geometry.maxSizes[0];
  if(e<8||e>384||in->inputTensors[2].geometry.maxSizes[0]!=6144||in->inputTensors[2].geometry.maxSizes[1]!=e||y.maxSizes[0]!=1536u*e||y.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=1;out->indexSpaceGeometry[0]=e*3;
  for(int j=0;j<5;++j)out->inputTensorAccessPattern[j].allRequired=1;
  map(o,0,0,512,0,511);map(o,1,0,0,0,0);
  out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/e);
#else
  int n=y.maxSizes[0],m=y.maxSizes[1],splits=y.maxSizes[2],e=y.maxSizes[3],k=in->inputTensors[3].geometry.maxSizes[0];
  if(n%GK_GROUPED_TILE||k%32||m!=GK_SMALLM_ROWS||e<8||e>384)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  int chunk=(k/32+splits-1)/splits;
  out->indexSpaceGeometry[0]=n/GK_GROUPED_TILE;out->indexSpaceGeometry[1]=e;out->indexSpaceGeometry[2]=splits;
  // Source expert is device data, not affine in the index-space expert slot.
  for(int j=0;j<3;++j)out->inputTensorAccessPattern[j].allRequired=1;
  auto&x=out->inputTensorAccessPattern[3];map(x,0,2,chunk*32,0,chunk*32-1);map(x,1,1,0,0,m-1);map(x,2,1,1,0,0);
  for(int j=4;j<6;++j)map(out->inputTensorAccessPattern[j],0,1,1,0,0);
  map(o,0,0,GK_GROUPED_TILE,0,GK_GROUPED_TILE-1);map(o,1,1,0,0,m-1);map(o,2,2,1,0,0);map(o,3,1,1,0,0);
#endif
 }else if(id==1){
  int n=y.maxSizes[0],k=y.maxSizes[1],e=y.maxSizes[2];if(n%256||k%32||e<1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=n/256;out->indexSpaceGeometry[1]=k/32;out->indexSpaceGeometry[2]=e;
  for(int j=0;j<3;++j)out->inputTensorAccessPattern[j].allRequired=1;
  map(out->inputTensorAccessPattern[3],0,2,1,0,0);
  map(o,0,0,256,0,255);map(o,1,1,32,0,31);map(o,2,2,1,0,0);
 }else if(id==3){
  const auto&x=in->inputTensors[0].geometry;
  if(y.maxSizes[0]!=x.maxSizes[0]||y.maxSizes[1]!=x.maxSizes[1]||y.maxSizes[2]!=x.maxSizes[2])return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=y.maxSizes[1];out->indexSpaceGeometry[2]=y.maxSizes[2];
  auto&xp=out->inputTensorAccessPattern[0];
  map(xp,0,0,128,0,127);map(xp,1,1,1,0,0);map(xp,2,2,1,0,0);
  map(o,0,0,128,0,127);map(o,1,1,1,0,0);map(o,2,2,1,0,0);
  for(int j=1;j<4;++j)out->inputTensorAccessPattern[j].allRequired=1;
 }else{
  int width=
#ifdef GK_LITERAL_GP
  128;
#else
  64;
#endif
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+width-1)/width;out->indexSpaceGeometry[1]=y.maxSizes[1];out->indexSpaceGeometry[2]=y.maxSizes[2];
  auto&x=out->inputTensorAccessPattern[0];map(x,0,0,width,0,width-1);map(x,1,1,1,0,0);map(x,2,0,0,0,in->inputTensors[0].geometry.maxSizes[2]-1);map(x,3,2,1,0,0);
  map(o,0,0,width,0,width-1);map(o,1,1,1,0,0);map(o,2,2,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
