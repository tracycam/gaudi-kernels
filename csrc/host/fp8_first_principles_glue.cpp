// Experimental offline-qualified kernels. Runtime qualification is pending.
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <climits>
using namespace tpc_lib_api;
#define KERNELS(X) X(amax16) X(lut_corrected) X(lut_single) X(lut_single4)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start, _binary_##n##_o_end;
KERNELS(DECL)
#undef DECL
static const char* names[]={"fp8_fp_quant_amax16","fp8_fp_quant_lut_corrected","fp8_fp_quant_lut_single","fp8_fp_quant_lut_single4"};
static int which(const char* name){for(int i=0;i<4;++i)if(!std::strcmp(name,names[i]))return i;return -1;}
template<class T> static void map(T& p,int dim,int index,float stride,int lo,int hi){
 p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t* count,GuidInfo* guids){
 *count=d==DEVICE_ID_GAUDI2?4:0;if(guids)for(unsigned i=0;i<*count;++i)std::strcpy(guids[i].name,names[i]);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=(id?2u:1u))return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->outputTensorNr!=2)return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
 const auto& x=in->inputTensors[0].geometry;const auto& q=in->outputTensors[0].geometry;
 const auto& s=in->outputTensors[1].geometry;
 if(x.dataType!=DATA_BF16||q.dataType!=DATA_F8_143||s.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_DATA_TYPE;
 if(x.dims!=2||x.maxSizes[0]==0||x.maxSizes[1]==0)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 // TPC width/row induction is signed i32, including the unrolled K+384 and
 // final K+=512 update. This experimental API supports bounded static shapes.
 if(x.maxSizes[0]>unsigned(INT_MAX-511)||x.maxSizes[1]>unsigned(INT_MAX))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(q.dims!=2||q.maxSizes[0]!=x.maxSizes[0]||q.maxSizes[1]!=x.maxSizes[1]||s.dims!=2||s.maxSizes[0]!=1||s.maxSizes[1]!=x.maxSizes[1])return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
 if(id){const auto& t=in->inputTensors[1].geometry;
  if(t.dataType!=DATA_F32)return GLUE_INCOMPATIBLE_DATA_TYPE;
  if(t.dims!=2||t.maxSizes[0]!=129||t.maxSizes[1]!=1)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 }
 out->indexSpaceRank=2;out->indexSpaceGeometry[0]=x.maxSizes[1];out->indexSpaceGeometry[1]=1;
 auto& a=out->inputTensorAccessPattern[0];a.allRequired=0;
 map(a,0,0,0,0,x.maxSizes[0]-1);map(a,1,0,1,0,0);
 if(id){auto& t=out->inputTensorAccessPattern[1];t.allRequired=1;map(t,0,0,0,0,128);map(t,1,0,0,0,0);}
 for(int i=0;i<2;++i){auto& o=out->outputTensorAccessPattern[i];o.allRequired=0;map(o,0,0,0,0,i?0:x.maxSizes[0]-1);map(o,1,0,1,0,0);}
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char* begin[]={KERNELS(BEGIN)},*end[]={KERNELS(END)};
 unsigned capacity=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.paramsNr=0;out->kernel.elfSize=size;
 if(capacity<size)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*){
 return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* layouts,uint32_t* count){
 if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned capacity=*count;*count=1;
 if(!layouts||!capacity)return GLUE_SUCCESS;
 for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(layouts[0].inputs[i].layout,'x',sizeof(layouts[0].inputs[i].layout));
 for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(layouts[0].outputs[i].layout,'x',sizeof(layouts[0].outputs[i].layout));return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
