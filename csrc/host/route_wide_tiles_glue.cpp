// PRIVATE wide-row experiment: copied from csrc/host/moe_route_tiles_glue.cpp at 822e049.
// Only namespace/GUID and explicit row limit change; original interface is untouched.
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
#define ELF(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELF(gather) ELF(gate) ELF(combine)
namespace {
const char*names[]={"gk_route_wide_tile_gather_bf16_v1","gk_route_wide_tile_gate_bf16_v1","gk_route_wide_tile_combine_f32_v1"};
int which(const char*s){for(int i=0;i<3;++i)if(!std::strcmp(s,names[i]))return i;return -1;}
bool shape(const Tensor&t,TensorDataType dtype,std::initializer_list<unsigned> dims){
 if(t.geometry.dataType!=dtype||t.geometry.dims!=dims.size())return false;
 unsigned d=0;for(auto n:dims){if(!n||t.geometry.minSizes[d]!=n||t.geometry.maxSizes[d]!=n)return false;++d;}return true;
}
template<class T>void map(T&p,int d,int i,int a,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=0;p.mapping[d].end_b=hi;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?3:0;if(g)for(unsigned i=0;i<std::min(cap,*n);++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 unsigned ni=id==2?4:3,np=id==0?3:id==1?1:0;
 if(in->inputTensorNr!=ni||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=np*4||(np&&!in->nodeParams.nodeParams))return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 int p[3]={};if(np)std::memcpy(p,in->nodeParams.nodeParams,np*4);
 for(unsigned i=0;i<ni+1;++i){const auto&g=(i<ni?in->inputTensors[i]:in->outputTensors[0]).geometry;
  if(g.dims<1||g.dims>3)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned d=0;d<g.dims;++d)if(!g.maxSizes[d]||g.maxSizes[d]>513*8*6144||g.minSizes[d]!=g.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 const auto&x=in->inputTensors[0];const auto&y=in->outputTensors[0];
 unsigned w=y.geometry.maxSizes[0],c=y.geometry.maxSizes[1],batch=y.geometry.maxSizes[2];bool ok=false;
 if(id==0){unsigned t=x.geometry.maxSizes[1],b=in->inputTensors[1].geometry.maxSizes[0];
  ok=p[0]>=1&&p[0]<=8&&p[1]>=1&&p[1]<=128&&p[2]>=0&&t>=1&&t<=513&&w==6144&&c==unsigned(p[1])&&batch>=1&&batch<=513*8&&b%c==0&&uint64_t(p[2])+batch<=b/c&&shape(x,DATA_BF16,{w,t})&&shape(in->inputTensors[1],DATA_I32,{b})&&shape(in->inputTensors[2],DATA_I32,{1})&&shape(y,DATA_BF16,{w,c,batch});
 }else if(id==1){unsigned b=in->inputTensors[1].geometry.maxSizes[0];
  ok=p[0]>=0&&w==256&&c>=1&&c<=128&&batch>=1&&batch<=513*8&&uint64_t(p[0])+batch<=b&&shape(x,DATA_F32,{512,c,batch})&&shape(in->inputTensors[1],DATA_I32,{b})&&shape(in->inputTensors[2],DATA_I32,{1})&&shape(y,DATA_BF16,{256,c,batch});
 }else{unsigned r=in->inputTensors[1].geometry.maxSizes[0],rows=x.geometry.maxSizes[1];
  ok=w>=512&&w<=6144&&w%512==0&&c>=1&&c<=513&&r>=1&&r<=8&&rows<=513*8*128&&shape(x,DATA_F32,{w,rows})&&shape(in->inputTensors[1],DATA_F32,{r,c})&&shape(in->inputTensors[2],DATA_I32,{r*c})&&shape(in->inputTensors[3],DATA_I32,{1})&&shape(y,DATA_F32,{w,c});
 }
 if(!ok)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<ni;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&o=out->outputTensorAccessPattern[0];unsigned vector=id==2?64:128;
 out->indexSpaceRank=id==2?2:3;out->indexSpaceGeometry[0]=w/vector;out->indexSpaceGeometry[1]=c;
 map(o,0,0,vector,vector-1);map(o,1,1,1,0);
#if defined(GK_ROUTE_GATHER_ROW_FIRST) && GK_ROUTE_GATHER_ROW_FIRST
 if(id==0){
  // Same tensor geometry. Prefer whole rows when distributing index-space work.
  out->indexSpaceGeometry[0]=c;out->indexSpaceGeometry[1]=w/vector;
  map(o,0,1,vector,vector-1);map(o,1,0,1,0);
 }
#endif
 if(id!=2){out->indexSpaceGeometry[2]=batch;map(o,2,2,1,0);}
 out->kernel.paramsNr=np;if(np)std::memcpy(out->kernel.scalarParams,p,np*4);
 unsigned char*starts[]={&_binary_gather_o_start,&_binary_gate_o_start,&_binary_combine_o_start};
 unsigned char*ends[]={&_binary_gather_o_end,&_binary_gate_o_end,&_binary_combine_o_end};
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=ends[id]-starts[id];if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,starts[id],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
