#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_gate_o_start,_binary_gate_o_end,_binary_combine_o_start,_binary_combine_o_end;
static const char*names[]={"gk_moe_pipeline_gate","gk_moe_pipeline_combine"};
static int which(const char*n){for(int i=0;i<2;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){unsigned cap=*n;*n=d==DEVICE_ID_GAUDI2?2:0;if(g)for(unsigned i=0;i<cap&&i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=3||in->outputTensorNr!=(id==0?2:1))return GLUE_INCOMPATIBLE_INPUT_COUNT;
 auto&y=in->outputTensors[0].geometry;out->indexSpaceRank=id==0?3:2;out->indexSpaceGeometry[0]=(y.maxSizes[0]+(id==0?127:63))/(id==0?128:64);out->indexSpaceGeometry[1]=y.maxSizes[1];if(id==0)out->indexSpaceGeometry[2]=y.maxSizes[2];
 for(unsigned j=0;j<in->inputTensorNr;++j)out->inputTensorAccessPattern[j].allRequired=1;
 for(unsigned j=0;j<in->outputTensorNr;++j)out->outputTensorAccessPattern[j].allRequired=1;
 out->kernel.paramsNr=0;auto*start=id==0?&_binary_gate_o_start:&_binary_combine_o_start;auto*end=id==0?&_binary_gate_o_end:&_binary_combine_o_end;unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(l&&cap){for(unsigned j=0;j<in->inputTensorNr;++j)std::memset(l[0].inputs[j].layout,'x',sizeof(l[0].inputs[j].layout));for(unsigned j=0;j<in->outputTensorNr;++j)std::memset(l[0].outputs[j].layout,'x',sizeof(l[0].outputs[j].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
