#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(mxfp4_decode) X(mxfp4_gemv) X(mxfp4_bias) X(mxfp4_reduce)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char* names[]={"gk_mxfp4_decode_bf16_v1","gk_mxfp4_gemv_f32_v1","gk_mxfp4_bias_f32_v1","gk_mxfp4_reduce_f32_v1"};
static int which(const char*n){for(int i=0;i<4;++i)if(!std::strcmp(n,names[i]))return i;return -1;}
template<class T> static void map(T&p,int dim,int index,float stride,int lo,int hi){p.mapping[dim].indexSpaceDim=index;p.mapping[dim].a=stride;p.mapping[dim].start_b=lo;p.mapping[dim].end_b=hi;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?4:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=which(in->guid.name);if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=(id==0?3:id==1?4:id==2?2:1)||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 const auto& y=in->outputTensors[0].geometry;
 out->indexSpaceRank=2;out->kernel.paramsNr=0;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=0;
 out->outputTensorAccessPattern[0].allRequired=0;
 if(id==0){
  if(in->nodeParams.nodeParamsSize!=4)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  int offset=*static_cast<const int*>(in->nodeParams.nodeParams);
  out->kernel.paramsNr=1;out->kernel.scalarParams[0]=offset;
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+255)/256;out->indexSpaceGeometry[1]=(y.maxSizes[1]+31)/32;
  auto&w=out->inputTensorAccessPattern[0];map(w,0,0,0,0,127);map(w,1,1,32,0,31);map(w,2,0,1,offset,offset);
  auto&s=out->inputTensorAccessPattern[1];map(s,0,0,0,0,255);map(s,1,1,1,0,0);map(s,2,0,1,offset,offset);
  out->inputTensorAccessPattern[2].allRequired=1;
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,256,0,255);map(o,1,1,32,0,31);
 }else if(id==1){
  int groups=(in->inputTensors[3].geometry.maxSizes[0]+31)/32,splits=y.maxSizes[2],chunk=(groups+splits-1)/splits;
  out->indexSpaceRank=3;out->indexSpaceGeometry[2]=splits;
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+255)/256;out->indexSpaceGeometry[1]=y.maxSizes[1];
#ifdef GK_SMALLM_ROWS
  if(y.maxSizes[1]!=GK_SMALLM_ROWS)return GLUE_INCOMPATIBLE_INPUT_SIZE;
  out->indexSpaceGeometry[1]=1;
#endif
#ifdef GK_SMALLM_VECTOR
  if(in->inputTensors[3].geometry.maxSizes[0]%32)return GLUE_INCOMPATIBLE_INPUT_SIZE;
#endif
  for(int j=0;j<2;++j){auto&w=out->inputTensorAccessPattern[j];map(w,0,0,0,0,j?255:127);map(w,1,2,j?chunk:chunk*32,0,(j?chunk:chunk*32)-1);map(w,2,0,1,0,0);}
  out->inputTensorAccessPattern[2].allRequired=1;
  auto&x=out->inputTensorAccessPattern[3];map(x,0,2,chunk*32,0,chunk*32-1);map(x,1,1,1,0,0);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,256,0,255);map(o,1,1,1,0,0);map(o,2,2,1,0,0);
#ifdef GK_SMALLM_ROWS
  map(x,1,1,GK_SMALLM_ROWS,0,GK_SMALLM_ROWS-1);
  map(o,1,1,GK_SMALLM_ROWS,0,GK_SMALLM_ROWS-1);
#endif
 }else{
  out->indexSpaceGeometry[0]=(y.maxSizes[0]+63)/64;out->indexSpaceGeometry[1]=y.maxSizes[1];
  auto&x=out->inputTensorAccessPattern[0];map(x,0,0,64,0,63);map(x,1,1,1,0,0);
  if(id==2){auto&b=out->inputTensorAccessPattern[1];map(b,0,0,64,0,63);map(b,1,1,0,0,0);}
  else map(x,2,0,0,0,in->inputTensors[0].geometry.maxSizes[2]-1);
  auto&o=out->outputTensorAccessPattern[0];map(o,0,0,64,0,63);map(o,1,1,1,0,0);
 }
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};
 unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;
 if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(!ls||!cap)return GLUE_SUCCESS;for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));for(unsigned i=0;i<in->outputTensorNr;++i)std::memset(ls[0].outputs[i].layout,'x',sizeof(ls[0].outputs[i].layout));return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
