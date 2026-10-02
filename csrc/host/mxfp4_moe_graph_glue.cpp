#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(graph_count) X(graph_plan) X(graph_gather) X(graph_decode) X(graph_gate_rows) X(graph_combine) X(graph_literal_gate)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"gk_mxfp4_graph_count","gk_mxfp4_graph_plan","gk_mxfp4_graph_gather","gk_mxfp4_graph_decode_historical","gk_mxfp4_graph_gate_rows","gk_mxfp4_graph_combine","gk_mxfp4_graph_literal_gate"};
static int which(const char*n){for(int i=0;i<7;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,int a,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=a;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?7:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;unsigned ni[]={1,2,4,4,3,5,2},no[]={2,2,1,1,1,1,1},bytes[]={0,8,8,12,4,4,4};
 if(in->inputTensorNr!=ni[id]||in->outputTensorNr!=no[id]||in->nodeParams.nodeParamsSize!=bytes[id])return GLUE_INCOMPATIBLE_INPUT_COUNT;
 out->kernel.paramsNr=bytes[id]/4;if(bytes[id])std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,bytes[id]);for(unsigned i=0;i<ni[id];++i)out->inputTensorAccessPattern[i].allRequired=1;
 const auto&y=in->outputTensors[0].geometry;auto&o=out->outputTensorAccessPattern[0];
 if(id==0){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[0];for(int i=0;i<2;++i){map(out->outputTensorAccessPattern[i],0,0,1,0,0);map(out->outputTensorAccessPattern[i],1,0,0,0,0);}}
 else if(id==1){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=in->outputTensors[1].geometry.maxSizes[0];o.allRequired=1;auto&m=out->outputTensorAccessPattern[1];map(m,0,0,1,0,0);map(m,1,0,0,0,0);}
 else if(id==2){out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[2];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,0,0,y.maxSizes[1]-1);map(o,2,0,1,0,0);}
 else if(id==3){if(y.maxSizes[0]%512||y.maxSizes[1]%32)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;out->indexSpaceRank=3;out->indexSpaceGeometry[0]=y.maxSizes[1]/32;out->indexSpaceGeometry[1]=y.maxSizes[0]/512;out->indexSpaceGeometry[2]=y.maxSizes[2];map(o,0,1,512,0,511);map(o,1,0,32,0,31);map(o,2,2,1,0,0);}
 else if(id==4){out->indexSpaceRank=3;out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=y.maxSizes[1];out->indexSpaceGeometry[2]=y.maxSizes[2];map(o,0,0,128,0,127);map(o,1,1,1,0,0);map(o,2,2,1,0,0);}
 else if(id==5){out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+63)/64;out->indexSpaceGeometry[1]=y.maxSizes[1];map(o,0,0,64,0,63);map(o,1,1,1,0,0);}
 else {out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[1];map(o,0,0,0,0,y.maxSizes[0]-1);map(o,1,0,1,0,0);}
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
