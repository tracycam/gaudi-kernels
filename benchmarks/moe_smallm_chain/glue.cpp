#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>
using namespace tpc_lib_api;
#ifdef GK_SMALLM_CAP8
#define ELFS(X) X(metadata) X(gp4) X(down4) X(gp8) X(down8) X(gate) X(combine)
#else
#define ELFS(X) X(metadata) X(gp4) X(down4) X(gate) X(combine)
#endif
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_smallm_route_metadata","gk_smallm_gp4","gk_smallm_down4","gk_smallm_gp8","gk_smallm_down8","gk_smallm_gate","gk_smallm_combine"};
static bool enabled(int i){
#ifdef GK_SMALLM_CAP8
 return true;
#else
 return i!=3&&i!=4;
#endif
}
static int which(const char*s){for(int i=0;i<7;++i)if(enabled(i)&&!std::strcmp(s,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int idx,int a,int hi){p.mapping[d].indexSpaceDim=idx;p.mapping[d].a=a;p.mapping[d].start_b=0;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=0;if(d==DEVICE_ID_GAUDI2)for(int i=0;i<7;++i)if(enabled(i)){if(g&&*n<cap)std::strcpy(g[*n].name,names[i]);++*n;}return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 const unsigned ni[]={1,8,8,8,8,2,3},no[]={4,1,1,1,1,1,1};
 if(in->inputTensorNr!=ni[id]||in->outputTensorNr!=no[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned j=0;j<ni[id];++j)out->inputTensorAccessPattern[j].allRequired=1;
 for(unsigned j=0;j<no[id];++j)out->outputTensorAccessPattern[j].allRequired=0;
 out->kernel.paramsNr=0;
 const auto&y=in->outputTensors[0].geometry;auto&o=out->outputTensorAccessPattern[0];
 if(id==0){
  if(in->nodeParams.nodeParamsSize!=4)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  int cap;std::memcpy(&cap,in->nodeParams.nodeParams,4);int length=in->inputTensors[0].geometry.maxSizes[0];
  if((cap!=4&&cap!=8)||length<8||length>64||length%8||in->inputTensors[0].geometry.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=1;out->indexSpaceGeometry[0]=length;out->kernel.paramsNr=1;out->kernel.scalarParams[0]=cap;
  map(o,0,0,1,0);map(out->outputTensorAccessPattern[1],0,0,1,0);
  map(out->outputTensorAccessPattern[2],0,0,0,cap-1);map(out->outputTensorAccessPattern[2],1,0,1,0);
  map(out->outputTensorAccessPattern[3],0,0,1,0);
#ifdef GK_SMALLM_STRIPE
  for(int j=0;j<3;++j)out->outputTensorAccessPattern[j].allRequired=1;
#endif
 }else if(id<=4){
  int cap=id<=2?4:8;bool down=id==2||id==4;int n=down?6144:512,splits=down?1:3;
#ifdef GK_SMALLM_DENSE
  if(y.dims!=3||y.dataType!=DATA_F32||y.maxSizes[0]!=unsigned(n)||y.maxSizes[1]!=unsigned(splits)||y.maxSizes[2]<8||y.maxSizes[2]>64)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  unsigned slots=y.maxSizes[2];o.allRequired=1; // Device row_map is a disjoint scatter permutation.
#else
  if(y.dims!=4||y.dataType!=DATA_F32||y.maxSizes[0]!=unsigned(n)||y.maxSizes[1]!=unsigned(cap)||y.maxSizes[2]!=unsigned(splits)||y.maxSizes[3]<8||y.maxSizes[3]>64)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  unsigned slots=y.maxSizes[3];
  map(o,0,0,256,255);map(o,1,1,0,cap-1);map(o,2,2,1,0);map(o,3,1,1,0);
#endif
  if(in->inputTensors[6].geometry.maxSizes[0]!=unsigned(cap)||in->inputTensors[6].geometry.maxSizes[1]!=slots||in->inputTensors[7].geometry.maxSizes[0]!=slots)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=3;out->indexSpaceGeometry[0]=n/256;out->indexSpaceGeometry[1]=slots;out->indexSpaceGeometry[2]=splits;
#ifdef GK_SMALLM_WIDE_M1
  out->indexSpaceGeometry[0]=n/512;
#endif
 }else if(id==5){
#ifdef GK_SMALLM_DENSE
  if(y.dims!=2||y.dataType!=DATA_BF16||y.maxSizes[0]!=256)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=2;out->indexSpaceGeometry[1]=y.maxSizes[1];
  map(o,0,0,128,127);map(o,1,1,1,0);
#else
  if(y.dims!=3||y.dataType!=DATA_BF16||y.maxSizes[0]!=256)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=3;out->indexSpaceGeometry[0]=2;out->indexSpaceGeometry[1]=y.maxSizes[1];out->indexSpaceGeometry[2]=y.maxSizes[2];
  map(o,0,0,128,127);map(o,1,1,1,0);map(o,2,2,1,0);
#endif
 }else{
  if(y.dims!=2||y.dataType!=DATA_F32||y.maxSizes[0]!=6144)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=96;out->indexSpaceGeometry[1]=y.maxSizes[1];
  map(o,0,0,64,63);map(o,1,1,1,0);
 }
#define START(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*starts[]={ELFS(START)},*ends[]={ELFS(END)};
#ifndef GK_SMALLM_CAP8
 if(id>=5)id-=2;
#endif
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=ends[id]-starts[id];if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,starts[id],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
