#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
#include <climits>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_vendor_o_start,_binary_vendor_o_end,_binary_fp32_o_start,_binary_fp32_o_end;
namespace {
const char*names[]={"gk_ag8_residual_norm_vendor_boundaries_experiment","gk_ag8_residual_norm_fp32_statistics_experiment"};
int which(const char*n){for(int i=0;i<2;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T>void map(T&p,int d,float step,int hi){p.mapping[d].indexSpaceDim=0;p.mapping[d].a=step;p.mapping[d].start_b=0;p.mapping[d].end_b=hi;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?2:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int kind=which(in->guid.name);if(kind<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=3)return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float eps=*static_cast<const float*>(in->nodeParams.nodeParams);if(!std::isfinite(eps)||eps<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 auto&r=in->inputTensors[1].geometry;auto h=r.maxSizes[0],m=r.maxSizes[1];
 if(r.dims!=2||!h||h>8192||!m||m>INT_MAX/8)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 for(unsigned i=0;i<3;++i){auto&t=in->inputTensors[i].geometry;
  if(t.dataType!=(i?DATA_BF16:DATA_F32))return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=(i==2?1u:2u)||t.maxSizes[0]!=h||(i!=2&&t.maxSizes[1]!=m*(i?1:8)))return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for(unsigned d=0;d<t.dims;++d)if(t.minSizes[d]!=t.maxSizes[d])return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 for(unsigned i=0;i<2;++i){auto&t=in->outputTensors[i].geometry;
  if(t.dataType!=DATA_BF16)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=2||t.maxSizes[0]!=h||t.maxSizes[1]!=m||t.minSizes[0]!=h||t.minSizes[1]!=m)return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 }
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=m;
 for(unsigned i=0;i<3;++i){auto&p=out->inputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,h-1);if(i!=2)map(p,1,1,i?0:7*m);}
 for(unsigned i=0;i<2;++i){auto&p=out->outputTensorAccessPattern[i];p.allRequired=0;map(p,0,0,h-1);map(p,1,1,0);}
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&eps,4);
 auto*start=kind?&_binary_fp32_o_start:&_binary_vendor_o_start;auto*end=kind?&_binary_fp32_o_end:&_binary_vendor_o_end;
 unsigned capacity=out->kernel.elfSize;out->kernel.elfSize=end-start;if(capacity<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
