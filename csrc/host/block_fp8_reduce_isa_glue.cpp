#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
#define ELFS(X) X(neumaier_baseline) X(neumaier_lookahead) X(neumaier_handschedule) X(neumaier_baseline_debug) X(neumaier_lookahead_debug) X(neumaier_handschedule_debug)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_neumaier_isa_baseline_v1","gk_neumaier_isa_lookahead_v1","gk_neumaier_isa_handschedule_v1","gk_neumaier_isa_baseline_debug_v1","gk_neumaier_isa_lookahead_debug_v1","gk_neumaier_isa_handschedule_debug_v1"};
static int which(const char*n){for(int i=0;i<6;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>static void map(T&p,int d,int i,float stride,int lo,int hi){p.mapping[d].indexSpaceDim=i;p.mapping[d].a=stride;p.mapping[d].start_b=lo;p.mapping[d].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?6:0;if(g)for(unsigned i=0;i<cap&&i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=4||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto&x=in->inputTensors[0].geometry;auto&y=in->outputTensors[0].geometry;
 for(unsigned i=0;i<4;++i){if(in->inputTensors[i].geometry.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_INPUT_SIZE;out->inputTensorAccessPattern[i].allRequired=0;}out->outputTensorAccessPattern[0].allRequired=0;
 if(y.dataType!=(id>=3?DATA_F32:DATA_BF16))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 const auto&as=in->inputTensors[1].geometry;const auto&ws=in->inputTensors[2].geometry;const auto&bgeo=in->inputTensors[3].geometry;
 if(as.dims!=3||as.maxSizes[0]!=1||as.maxSizes[1]!=x.maxSizes[1]||as.maxSizes[2]!=x.maxSizes[2]||ws.dims!=2||ws.maxSizes[0]!=x.maxSizes[2]||ws.maxSizes[1]!=(x.maxSizes[0]+127)/128||bgeo.dims!=1||bgeo.maxSizes[0]!=x.maxSizes[0])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(x.dims!=3||y.dims!=2||x.maxSizes[0]!=y.maxSizes[0]||x.maxSizes[1]!=y.maxSizes[1])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<4;++i)for(unsigned d=0;d<in->inputTensors[i].geometry.dims;++d)if(!in->inputTensors[i].geometry.maxSizes[d]||in->inputTensors[i].geometry.maxSizes[d]>INT32_MAX-128)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 unsigned rows=1,g=x.maxSizes[2];out->indexSpaceRank=2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+127)/128;out->indexSpaceGeometry[1]=(y.maxSizes[1]+rows-1)/rows;
 auto&p=out->inputTensorAccessPattern[0];map(p,0,0,128,0,127);map(p,1,1,rows,0,rows-1);map(p,2,0,0,0,g-1);
 auto&a=out->inputTensorAccessPattern[1];map(a,0,0,0,0,0);map(a,1,1,rows,0,rows-1);map(a,2,0,0,0,g-1);
 auto&w=out->inputTensorAccessPattern[2];map(w,0,0,0,0,g-1);map(w,1,0,1,0,0);
 auto&b=out->inputTensorAccessPattern[3];map(b,0,0,128,0,127);
 auto&o=out->outputTensorAccessPattern[0];map(o,0,0,128,0,127);map(o,1,1,rows,0,rows-1);
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.paramsNr=0;out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}

extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
