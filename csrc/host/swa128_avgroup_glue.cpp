// Offline-only candidate; content validity of dynamic page tables is caller-owned.
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cmath>
#include <climits>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_quad_o_start,_binary_quad_o_end,_binary_group_o_start,_binary_group_o_end,_binary_quad_debug_o_start,_binary_quad_debug_o_end,_binary_group_debug_o_start,_binary_group_debug_o_end;
static const char*guids[]={"gk_swa128_avgroup_quad_v0","gk_swa128_avgroup_group_v0","gk_swa128_avgroup_quad_debug_v0","gk_swa128_avgroup_group_debug_v0"};
static int kind(const char*n){for(int i=0;i<4;++i)if(!std::strcmp(n,guids[i]))return i;return -1;}
static bool shape(const Tensor&t,TensorDataType type,std::initializer_list<unsigned long>dims){
 if(t.geometry.dataType!=type||t.geometry.dims!=dims.size())return false;
 unsigned i=0;for(auto v:dims){if(t.geometry.maxSizes[i]!=v||t.geometry.minSizes[i]!=v)return false;++i;}return true;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*c,GuidInfo*g){*c=d==DEVICE_ID_GAUDI2?4:0;if(g)for(unsigned i=0;i<*c;++i)std::strcpy(g[i].name,guids[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int variant=kind(in->guid.name);if(variant<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=7)return GLUE_INCOMPATIBLE_INPUT_COUNT;if(in->outputTensorNr!=(variant>=2?4:1))return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=4||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 float scale=*static_cast<float*>(in->nodeParams.nodeParams);if(!std::isfinite(scale)||scale<=0)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 bool flat=in->inputTensors[0].geometry.maxSizes[0]!=192;
 unsigned long heads=flat?in->inputTensors[0].geometry.maxSizes[0]/192:in->inputTensors[0].geometry.maxSizes[1],slots=in->inputTensors[1].geometry.maxSizes[2];
 if(!heads||heads>16||!slots||slots%128||slots>INT_MAX/192)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!(flat?shape(in->inputTensors[0],DATA_BF16,{heads*192,1,1}):shape(in->inputTensors[0],DATA_BF16,{192,heads,1}))||!shape(in->inputTensors[1],DATA_BF16,{192,1,slots})||!shape(in->inputTensors[2],DATA_BF16,{128,1,slots})||!shape(in->inputTensors[6],DATA_BF16,{heads}))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 unsigned long pages=in->inputTensors[3].geometry.maxSizes[0];
 if(!pages||pages>INT_MAX||!shape(in->inputTensors[3],DATA_I32,{pages})||!shape(in->inputTensors[4],DATA_I32,{pages})||!(shape(in->inputTensors[5],DATA_I32,{1})||shape(in->inputTensors[5],DATA_I32,{1,1})))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(!shape(in->outputTensors[0],DATA_BF16,{128,heads,1}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 if(variant>=2)for(unsigned i=1;i<4;++i)if(!shape(in->outputTensors[i],DATA_F32,{128,heads,1}))return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=heads;
 // Dynamic physical pages are conservatively allRequired. This descriptor does
 // not claim the whole cache is physically fetched; ISA guards each actual load.
 for(unsigned i=0;i<7;++i)out->inputTensorAccessPattern[i].allRequired=1;
 auto&p=out->outputTensorAccessPattern[0];p.allRequired=0;
 p.mapping[0]={0,0,0,127};p.mapping[1]={0,1,0,0};p.mapping[2]={0,0,0,0};
 if(variant>=2)for(unsigned i=1;i<4;++i)out->outputTensorAccessPattern[i]=out->outputTensorAccessPattern[0];
 out->kernel.paramsNr=1;std::memcpy(out->kernel.scalarParams,&scale,4);unsigned cap=out->kernel.elfSize;unsigned char*starts[]={&_binary_quad_o_start,&_binary_group_o_start,&_binary_quad_debug_o_start,&_binary_group_debug_o_start};unsigned char*ends[]={&_binary_quad_o_end,&_binary_group_o_end,&_binary_quad_debug_o_end,&_binary_group_debug_o_end};auto*start=starts[variant];auto*end=ends[variant];out->kernel.elfSize=end-start;
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,start,out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return kind(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*l,uint32_t*c){if(kind(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;auto cap=*c;*c=1;if(!l||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(l[0].inputs[i].layout,'x',sizeof(l[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(l[0].outputs[i].layout,'x',sizeof(l[0].outputs[i].layout));return GLUE_SUCCESS;}
