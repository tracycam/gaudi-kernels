// PRIVATE wide-row experiment: copied from csrc/host/moe_route_metadata_v3_glue.cpp at 822e049.
// Only namespace/GUID and explicit row limit change; original interface is untouched.
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>
using namespace tpc_lib_api;
#define ELF(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELF(count) ELF(prefix) ELF(inverse) ELF(row_map)
namespace {
const char*names[]={"gk_route_wide_count_i32_v3","gk_route_wide_prefix_i32_v3","gk_route_wide_inverse_i32_v3","gk_route_wide_row_map_i32_v3"};
int which(const char*s){for(int i=0;i<4;++i)if(!std::strcmp(s,names[i]))return i;return -1;}
bool vec(const Tensor&t,unsigned n){return t.geometry.dims==1&&t.geometry.maxSizes[0]==n&&t.geometry.minSizes[0]==n;}
bool matrix(const Tensor&t,unsigned n,unsigned m){return t.geometry.dims==2&&t.geometry.maxSizes[0]==n&&t.geometry.maxSizes[1]==m&&t.geometry.minSizes[0]==n&&t.geometry.minSizes[1]==m;}
bool domain(int t,int r,int e,int c,int b){return t>=1&&t<=513&&r>=1&&r<=8&&e>=r&&e<=384&&c>=1&&c<=128&&c<=t&&b>=1&&b<=t*r;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?4:0;if(g)for(unsigned i=0;i<std::min(cap,*n);++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;const unsigned ni[]={2,2,5,3},no[]={3,5,1,1},np[]={1,3,2,1};
 if(in->inputTensorNr!=ni[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=no[id])return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 for(unsigned i=0;i<ni[id]+no[id];++i){auto&g=(i<ni[id]?in->inputTensors[i]:in->outputTensors[i-ni[id]]).geometry;if(g.dataType!=DATA_I32)return GLUE_INCOMPATIBLE_DATA_TYPE;if(g.dims<1||g.dims>2)return GLUE_INCOMPATIBLE_INPUT_SIZE;for(unsigned d=0;d<g.dims;++d)if(!g.maxSizes[d]||g.maxSizes[d]>513*8*128||g.minSizes[d]!=g.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;}
 if(!in->nodeParams.nodeParams||in->nodeParams.nodeParamsSize!=np[id]*4)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;int p[3]={};std::memcpy(p,in->nodeParams.nodeParams,np[id]*4);
 auto&x=in->inputTensors[0].geometry;int tasks=0;bool good=false;
 if(id==3){int length=x.maxSizes[0],b=in->inputTensors[1].geometry.maxSizes[0],c=p[0];good=c>=1&&c<=128&&length>=1&&length<=513*8&&b>=1&&b<=length&&vec(in->inputTensors[0],length)&&vec(in->inputTensors[1],b)&&vec(in->inputTensors[2],1)&&vec(in->outputTensors[0],b*c);tasks=std::max(length,b);}
 else{int t=id==1?in->inputTensors[1].geometry.maxSizes[0]:x.maxSizes[1],r=id==1?p[0]:x.maxSizes[0],e=id==0?p[0]:id==1?x.maxSizes[0]:in->inputTensors[2].geometry.maxSizes[0]-1,c=id==0?1:p[id==1?1:0],b=id==0?1:p[id==1?2:1];if(!domain(t,r,e,c,b))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if(id==0)good=matrix(in->inputTensors[0],r,t)&&vec(in->inputTensors[1],t*r)&&vec(in->outputTensors[0],e)&&vec(in->outputTensors[1],t)&&matrix(in->outputTensors[2],(t*r+63)/64+1,e);
  if(id==1)good=vec(in->inputTensors[0],e)&&vec(in->inputTensors[1],t)&&vec(in->outputTensors[0],e+1)&&vec(in->outputTensors[1],b)&&vec(in->outputTensors[2],b)&&vec(in->outputTensors[3],b)&&vec(in->outputTensors[4],1);
  if(id==2)good=matrix(in->inputTensors[0],r,t)&&vec(in->inputTensors[1],t*r)&&vec(in->inputTensors[2],e+1)&&vec(in->inputTensors[3],1)&&matrix(in->inputTensors[4],(t*r+63)/64+1,e)&&vec(in->outputTensors[0],t*r);
  tasks=id==0?std::max(t,e):id==1?1:t*r;
 }
 if(!good)return GLUE_INCOMPATIBLE_INPUT_SIZE;out->indexSpaceRank=1;out->indexSpaceGeometry[0]=tasks;for(unsigned i=0;i<ni[id];++i)out->inputTensorAccessPattern[i].allRequired=1;for(unsigned i=0;i<no[id];++i)out->outputTensorAccessPattern[i].allRequired=1;
 out->kernel.paramsNr=np[id];std::memcpy(out->kernel.scalarParams,p,np[id]*4);unsigned char*start[]={&_binary_count_o_start,&_binary_prefix_o_start,&_binary_inverse_o_start,&_binary_row_map_o_start};unsigned char*end[]={&_binary_count_o_end,&_binary_prefix_o_end,&_binary_inverse_o_end,&_binary_row_map_o_end};unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end[id]-start[id];if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start[id],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
