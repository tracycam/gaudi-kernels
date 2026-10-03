#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <cstring>
#include <initializer_list>
using namespace tpc_lib_api;
#define ELF(n) extern "C" unsigned char _binary_pc_##n##_o_start,_binary_pc_##n##_o_end;
ELF(prefix) ELF(inverse) ELF(row_map) ELF(gather) ELF(gate) ELF(combine) ELF(decode) ELF(queue) ELF(queue_gemv)
namespace {
const char* names[]={"gk_expert_prefix","gk_expert_inverse","gk_expert_map","gk_expert_gather","gk_expert_gate","gk_expert_combine","gk_expert_decode_k8","gk_expert_queue","gk_expert_queue_gemv"};
int which(const char*s){for(int i=0;i<9;++i)if(!std::strcmp(s,names[i]))return i;return -1;}
bool shape(const Tensor&t,TensorDataType dt,std::initializer_list<unsigned> sizes){
 if(t.geometry.dataType!=dt||t.geometry.dims!=sizes.size())return false;
 unsigned i=0;for(auto n:sizes){if(!n||t.geometry.maxSizes[i]!=n||t.geometry.minSizes[i]!=n)return false;++i;}return true;
}
template<class T>void map(T&p,int d,int i,int a,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=0;p.mapping[d].end_b=hi;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?9:0;if(g)for(unsigned i=0;i<std::min(cap,*n);++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 const unsigned ni[]={2,5,3,3,3,4,4,1,6},no[]={5,1,1,1,1,1,1,2,1},np[]={6,2,1,3,1,0,4,0,0};
 if(in->inputTensorNr!=ni[id]||in->outputTensorNr!=no[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=np[id]*4||(np[id]&&!in->nodeParams.nodeParams))return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 int p[6]={};if(np[id])std::memcpy(p,in->nodeParams.nodeParams,np[id]*4);
 for(unsigned i=0;i<ni[id]+no[id];++i){auto&g=(i<ni[id]?in->inputTensors[i]:in->outputTensors[i-ni[id]]).geometry;
  if(g.dims<1||g.dims>3)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned d=0;d<g.dims;++d)if(!g.maxSizes[d]||g.maxSizes[d]>513ull*8*6144||g.minSizes[d]!=g.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 auto*x=in->inputTensors;auto*y=in->outputTensors;bool good=false;unsigned tasks=0;
 if(id==0){unsigned e=x[0].geometry.maxSizes[0],t=x[1].geometry.maxSizes[0];int r=p[0],c=p[1],b=p[2];
  good=e<=384&&t<=513&&r>=1&&r<=8&&e>=unsigned(r)&&c>=1&&c<=int(t)&&b>=1&&b<=int(e)&&p[3]>=1&&p[3]<=p[4]&&p[4]<=c&&p[5]>=0&&p[5]<=1&&shape(x[0],DATA_I32,{e})&&shape(x[1],DATA_I32,{t})&&shape(y[0],DATA_I32,{e+1})&&shape(y[1],DATA_I32,{unsigned(b)})&&shape(y[2],DATA_I32,{unsigned(b)})&&shape(y[3],DATA_I32,{unsigned(b)})&&shape(y[4],DATA_I32,{1});tasks=1;
 }else if(id==1){unsigned r=x[0].geometry.maxSizes[0],t=x[0].geometry.maxSizes[1],e=x[2].geometry.maxSizes[0]-1,c=p[0],b=p[1];
  good=t<=513&&r<=8&&e>=r&&e<=384&&c>=1&&c<=t&&b>=1&&b<=e&&shape(x[0],DATA_I32,{r,t})&&shape(x[1],DATA_I32,{r*t})&&shape(x[2],DATA_I32,{e+1})&&shape(x[3],DATA_I32,{1})&&shape(x[4],DATA_I32,{(r*t+63)/64+1,e})&&shape(y[0],DATA_I32,{r*t});tasks=r*t;
 }else if(id==2){unsigned length=x[0].geometry.maxSizes[0],b=x[1].geometry.maxSizes[0],c=p[0];
  good=length<=513*8&&b<=384&&c>=1&&c<=513&&shape(x[0],DATA_I32,{length})&&shape(x[1],DATA_I32,{b})&&shape(x[2],DATA_I32,{1})&&shape(y[0],DATA_I32,{b*c});tasks=std::max(length,b);
 }else if(id==3){unsigned t=x[0].geometry.maxSizes[1],length=x[1].geometry.maxSizes[0],c=y[0].geometry.maxSizes[1],b=y[0].geometry.maxSizes[2];
  good=t<=513&&p[0]>=1&&p[0]<=8&&p[1]>=1&&p[1]<=int(t)&&p[2]>=0&&c==unsigned(p[1])&&length%c==0&&uint64_t(p[2])+b<=length/c&&shape(x[0],DATA_BF16,{6144,t})&&shape(x[1],DATA_I32,{length})&&shape(x[2],DATA_I32,{1})&&shape(y[0],DATA_BF16,{6144,c,b});
 }else if(id==4){unsigned c=y[0].geometry.maxSizes[1],b=y[0].geometry.maxSizes[2],slots=x[1].geometry.maxSizes[0];
  good=c>=1&&c<=513&&p[0]>=0&&uint64_t(p[0])+b<=slots&&shape(x[0],DATA_F32,{512,c,b})&&shape(x[1],DATA_I32,{slots})&&shape(x[2],DATA_I32,{1})&&shape(y[0],DATA_BF16,{256,c,b});
 }else if(id==5){unsigned w=y[0].geometry.maxSizes[0],t=y[0].geometry.maxSizes[1],r=x[1].geometry.maxSizes[0],rows=x[0].geometry.maxSizes[1];
  good=w>=512&&w<=6144&&w%512==0&&t<=513&&r<=8&&shape(x[0],DATA_F32,{w,rows})&&shape(x[1],DATA_F32,{r,t})&&shape(x[2],DATA_I32,{r*t})&&shape(x[3],DATA_I32,{1})&&shape(y[0],DATA_F32,{w,t});
 }
 if(id==6){unsigned nt=y[0].geometry.maxSizes[0],k=y[0].geometry.maxSizes[1],batch=y[0].geometry.maxSizes[2];
  unsigned slots=x[3].geometry.maxSizes[0],owners=x[0].geometry.maxSizes[1];
  good=((k==6144&&p[0]==1)||(k==256&&p[0]==12))&&nt>=512&&nt%512==0&&p[1]>=0&&p[2]>=0&&p[2]+nt/512<=unsigned(p[0])&&uint64_t(p[1])+batch<=slots&&batch>=1&&uint64_t(batch)*k*nt*2<=16*1024*1024&&p[3]>=0&&p[3]<=2&&owners%(k*p[0])==0&&owners/(k*p[0])<=384&&shape(x[0],DATA_U8,{256,owners})&&shape(x[1],DATA_U8,{512,owners/32})&&shape(x[2],DATA_BF16,{512,1})&&shape(x[3],DATA_I32,{slots,1})&&shape(y[0],DATA_BF16,{nt,k,batch});
 }
 if(id==7){unsigned t=x[0].geometry.maxSizes[1],r=x[0].geometry.maxSizes[0];
  good=t<=513&&r>=1&&r<=8&&shape(x[0],DATA_I32,{r,t})&&shape(y[0],DATA_I32,{3*t*r})&&shape(y[1],DATA_I32,{12*t*r});tasks=1;
 }
 if(id==8){unsigned k=x[2].geometry.maxSizes[0],n=x[2].geometry.maxSizes[1],w=x[0].geometry.maxSizes[1];
  good=(k==2048||k==256)&&shape(x[0],DATA_U8,{256,w})&&shape(x[1],DATA_U8,{512,w/32})&&shape(x[2],DATA_BF16,{k,n})&&shape(x[3],DATA_BF16,{512,1})&&shape(x[4],DATA_I32,{n,1})&&shape(x[5],DATA_I32,{n})&&shape(y[0],DATA_F32,{512*n,1});tasks=std::min(24u,n);
 }
 if(!good)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<ni[id];++i)out->inputTensorAccessPattern[i].allRequired=1;
 if(id<3||id>=7){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=tasks;for(unsigned i=0;i<no[id];++i)out->outputTensorAccessPattern[i].allRequired=1;}
 else if(id==6){out->indexSpaceRank=3;auto&o=out->outputTensorAccessPattern[0];
  out->indexSpaceGeometry[0]=y[0].geometry.maxSizes[0]/256;out->indexSpaceGeometry[1]=y[0].geometry.maxSizes[1]/32;out->indexSpaceGeometry[2]=y[0].geometry.maxSizes[2];
  map(o,0,0,256,255);map(o,1,1,32,31);map(o,2,2,1,0);
 }
 else{unsigned w=y[0].geometry.maxSizes[0],c=y[0].geometry.maxSizes[1],b=y[0].geometry.maxSizes[2],v=id==5?64:128;
  auto&o=out->outputTensorAccessPattern[0];out->indexSpaceRank=id==5?2:3;
  out->indexSpaceGeometry[0]=id==3?c:w/v;out->indexSpaceGeometry[1]=id==3?w/v:c;
  map(o,0,id==3?1:0,v,v-1);map(o,1,id==3?0:1,1,0);
  if(id!=5){out->indexSpaceGeometry[2]=b;map(o,2,2,1,0);}
 }
 out->kernel.paramsNr=np[id];if(np[id])std::memcpy(out->kernel.scalarParams,p,np[id]*4);
#define START(n) &_binary_pc_##n##_o_start
#define END(n) &_binary_pc_##n##_o_end
 unsigned char*starts[]={START(prefix),START(inverse),START(row_map),START(gather),START(gate),START(combine),START(decode),START(queue),START(queue_gemv)};
 unsigned char*ends[]={END(prefix),END(inverse),END(row_map),END(gather),END(gate),END(combine),END(decode),END(queue),END(queue_gemv)};
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=ends[id]-starts[id];if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,starts[id],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
