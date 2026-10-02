#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_prepare_o_start,_binary_prepare_o_end;
extern "C" unsigned char _binary_fused_o_start,_binary_fused_o_end;
extern "C" unsigned char _binary_bounded_o_start,_binary_bounded_o_end;
extern "C" unsigned char _binary_lean_o_start,_binary_lean_o_end;
extern "C" unsigned char _binary_fill_o_start,_binary_fill_o_end;
extern "C" unsigned char _binary_gather_o_start,_binary_gather_o_end;
static const char*gather="gk_mixed_m_gather";
static const char*lean="gk_mixed_m_prepare_lean";
static const char*fill="gk_mixed_m_fill";
static const char*name="gk_mixed_m_prepare";
static const char*fused="gk_mixed_m_decode_prepare";
static const char*bounded="gk_mixed_m_prepare_bounded";
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?6:0;if(g&&*n){std::strcpy(g[0].name,name);std::strcpy(g[1].name,fused);std::strcpy(g[2].name,bounded);std::strcpy(g[3].name,lean);std::strcpy(g[4].name,fill);std::strcpy(g[5].name,gather);}return GLUE_SUCCESS;}
template<class T>static void map(T&p,int dim,int idx,float a,int hi){p.mapping[dim].indexSpaceDim=idx;p.mapping[dim].a=a;p.mapping[dim].start_b=0;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 if(!std::strcmp(in->guid.name,gather)){
  if(in->inputTensorNr!=3||in->outputTensorNr!=2||in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_COUNT;
  out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(in->inputTensors[0].geometry.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=in->inputTensors[2].geometry.maxSizes[0];
  out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
  for(int j=0;j<3;++j)out->inputTensorAccessPattern[j].allRequired=1;
  for(int j=0;j<2;++j)out->outputTensorAccessPattern[j].allRequired=1;
  unsigned cap=out->kernel.elfSize;out->kernel.elfSize=&_binary_gather_o_end-&_binary_gather_o_start;if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,&_binary_gather_o_start,out->kernel.elfSize);return GLUE_SUCCESS;
 }
 if(!std::strcmp(in->guid.name,lean)||!std::strcmp(in->guid.name,fill)){
  bool isfill=!std::strcmp(in->guid.name,fill);
  if(in->inputTensorNr!=(isfill?4:2)||in->outputTensorNr!=(isfill?1:2))return GLUE_INCOMPATIBLE_INPUT_COUNT;
  out->kernel.paramsNr=0;auto&y=in->outputTensors[0].geometry;
  if(isfill){
   out->indexSpaceRank=3;out->indexSpaceGeometry[0]=y.maxSizes[0]/256;out->indexSpaceGeometry[1]=y.maxSizes[1]/32;out->indexSpaceGeometry[2]=y.maxSizes[2];
   for(int j=0;j<3;++j)out->inputTensorAccessPattern[j].allRequired=1;
   map(out->inputTensorAccessPattern[3],0,2,1,0);auto&w=out->outputTensorAccessPattern[0];map(w,0,0,256,255);map(w,1,1,32,31);map(w,2,2,1,0);
  }else{
   auto&x=in->inputTensors[0].geometry;if(x.maxSizes[1]!=3||y.maxSizes[0]!=x.maxSizes[0]||y.maxSizes[1]!=3*x.maxSizes[2])return GLUE_INCOMPATIBLE_INPUT_SIZE;
   out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(x.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=x.maxSizes[2];
   for(int j=0;j<2;++j){out->inputTensorAccessPattern[j].allRequired=1;out->outputTensorAccessPattern[j].allRequired=1;}
  }
  auto*start=isfill?&_binary_fill_o_start:&_binary_lean_o_start;auto*end=isfill?&_binary_fill_o_end:&_binary_lean_o_end;unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
 }
 if(!std::strcmp(in->guid.name,fused)){
  if(in->inputTensorNr!=6||in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const auto&y=in->outputTensors[0].geometry;const auto&a=in->outputTensors[1].geometry;
  int n=y.maxSizes[0],k=y.maxSizes[1],e=y.maxSizes[2];
  if(n%256||k%32||a.maxSizes[0]!=unsigned(k)||a.maxSizes[1]!=3||a.maxSizes[2]!=unsigned(e))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->kernel.paramsNr=0;out->indexSpaceRank=3;out->indexSpaceGeometry[0]=n/256;out->indexSpaceGeometry[1]=k/32;out->indexSpaceGeometry[2]=e;
  for(int j=0;j<3;++j)out->inputTensorAccessPattern[j].allRequired=1;
  map(out->inputTensorAccessPattern[3],0,2,1,0);out->inputTensorAccessPattern[4].allRequired=1;map(out->inputTensorAccessPattern[5],0,2,1,0);
  auto&w=out->outputTensorAccessPattern[0];map(w,0,0,256,255);map(w,1,1,32,31);map(w,2,2,1,0);
  auto&ap=out->outputTensorAccessPattern[1];map(ap,0,1,32,127);map(ap,1,2,0,2);map(ap,2,2,1,0);
  unsigned cap=out->kernel.elfSize;out->kernel.elfSize=&_binary_fused_o_end-&_binary_fused_o_start;
  if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,&_binary_fused_o_start,out->kernel.elfSize);return GLUE_SUCCESS;
 }
 bool fast=!std::strcmp(in->guid.name,bounded);
 if(std::strcmp(in->guid.name,name)&&!fast)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=4||in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 const auto&x=in->inputTensors[0].geometry;int e=in->inputTensors[1].geometry.maxSizes[0];
 if(x.maxSizes[1]!=3||x.maxSizes[2]!=unsigned(e)||in->inputTensors[2].geometry.maxSizes[0]!=unsigned(e)||in->inputTensors[3].geometry.maxSizes[0]!=unsigned(e))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=e;
 if(fast){out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(x.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=e;}
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,4);
 // Dynamic scatter is one indivisible preparation node. Every output has
 // disjoint runtime ownership; an affine slice would be incorrect here.
 for(int i=0;i<4;++i)out->inputTensorAccessPattern[i].allRequired=1;
 for(int i=0;i<3;++i)out->outputTensorAccessPattern[i].allRequired=1;
 auto*start=fast?&_binary_bounded_o_start:&_binary_prepare_o_start;auto*end=fast?&_binary_bounded_o_end:&_binary_prepare_o_end;
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){const char*s=in->pGuid?in->pGuid->name:in->guid.name;return std::strcmp(s,name)&&std::strcmp(s,fused)&&std::strcmp(s,bounded)&&std::strcmp(s,lean)&&std::strcmp(s,fill)&&std::strcmp(s,gather)?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(std::strcmp(in->guid.name,name)&&std::strcmp(in->guid.name,fused)&&std::strcmp(in->guid.name,bounded)&&std::strcmp(in->guid.name,lean)&&std::strcmp(in->guid.name,fill)&&std::strcmp(in->guid.name,gather))return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(l&&cap){for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
