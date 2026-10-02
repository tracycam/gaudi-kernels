#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_scalar_o_start,_binary_scalar_o_end,_binary_vector_o_start,_binary_vector_o_end;
namespace {
struct Params{float factor;int renormalize;int apply_scale;};
const char*names[]={"gk_router_post8_scalar_experiment","gk_router_post8_vector_experiment"};
int which(const char*n){for(int i=0;i<2;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?2:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int kind=which(in->guid.name);if(kind<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=2)return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=3)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=sizeof(Params)||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 auto p=*static_cast<const Params*>(in->nodeParams.nodeParams);if(!std::isfinite(p.factor)||(p.renormalize!=0&&p.renormalize!=1)||(p.apply_scale!=0&&p.apply_scale!=1))return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 for(unsigned i=0;i<5;++i){auto&t=(i<2?in->inputTensors[i]:in->outputTensors[i-2]).geometry;
  auto type=i==1||i==3?DATA_I32:DATA_F32;unsigned width=i==0?384:i==4?1:8;
  if(t.dataType!=type)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=2||t.maxSizes[0]!=width||t.maxSizes[1]!=1||t.minSizes[0]!=width||t.minSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=1;for(unsigned i=0;i<2;++i)out->inputTensorAccessPattern[i].allRequired=1;for(unsigned i=0;i<3;++i)out->outputTensorAccessPattern[i].allRequired=1;
 out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,&p,sizeof(p));
 auto*start=kind?&_binary_vector_o_start:&_binary_scalar_o_start;auto*end=kind?&_binary_vector_o_end:&_binary_scalar_o_end;unsigned cap=out->kernel.elfSize;out->kernel.elfSize=end-start;if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
