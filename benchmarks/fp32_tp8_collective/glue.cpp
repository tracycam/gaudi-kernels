#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_sum_f32_o_start,_binary_sum_f32_o_end;
extern "C" unsigned char _binary_gate_f32_o_start,_binary_gate_f32_o_end;
static const char*guid="gk_probe_tp8_sum_f32";
static const char*gate_guid="gk_probe_tp8_gate_f32";
static bool known(const char*n){return !std::strcmp(n,guid)||!std::strcmp(n,gate_guid);}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){auto capacity=*n;*n=d==DEVICE_ID_GAUDI2?2:0;if(g&&capacity>=*n&&*n){std::strcpy(g[0].name,guid);std::strcpy(g[1].name,gate_guid);}return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*p,HabanaKernelInstantiation*q){
 if(!known(p->guid.name))return GLUE_NODE_NOT_FOUND;
 bool gate=!std::strcmp(p->guid.name,gate_guid);
 if(p->inputTensorNr!=1||p->outputTensorNr!=1||p->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 int ranks;std::memcpy(&ranks,p->nodeParams.nodeParams,4);auto&x=p->inputTensors[0].geometry;auto&y=p->outputTensors[0].geometry;
 if(x.dataType!=DATA_F32||y.dataType!=DATA_F32||x.maxSizes[0]!=y.maxSizes[0])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(gate){if(ranks<1||ranks>150000000||x.maxSizes[0]!=64||y.maxSizes[1]!=1||x.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;}
 else if((ranks!=1&&ranks!=8)||x.maxSizes[1]!=y.maxSizes[1]*ranks||y.maxSizes[0]%128)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 q->indexSpaceRank=1;q->indexSpaceGeometry[0]=gate?1:y.maxSizes[0]*y.maxSizes[1]/128;q->inputTensorAccessPattern[0].allRequired=1;q->outputTensorAccessPattern[0].allRequired=1;q->kernel.paramsNr=1;q->kernel.scalarParams[0]=ranks;
 auto*s=gate?&_binary_gate_f32_o_start:&_binary_sum_f32_o_start;auto*e=gate?&_binary_gate_f32_o_end:&_binary_sum_f32_o_end;
 unsigned n=e-s,cap=q->kernel.elfSize;q->kernel.elfSize=n;if(cap<n)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(q->kernel.kernelElf,s,n);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*p,ShapeInferenceOutput*){return known(p->pGuid?p->pGuid->name:p->guid.name)?GLUE_SUCCESS:GLUE_NODE_NOT_FOUND;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*p,NodeDataLayouts*l,uint32_t*n){unsigned cap=*n;*n=1;if(l&&cap){std::memset(l->inputs[0].layout,'x',sizeof(l->inputs[0].layout));std::memset(l->outputs[0].layout,'x',sizeof(l->outputs[0].layout));}return known(p->guid.name)?GLUE_SUCCESS:GLUE_NODE_NOT_FOUND;}
extern "C" uint64_t GetLibVersion(){return 1;}
