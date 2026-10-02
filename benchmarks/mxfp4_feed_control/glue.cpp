#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
extern "C" unsigned char _binary_bf16_o_start, _binary_bf16_o_end;
extern "C" unsigned char _binary_fp8_o_start, _binary_fp8_o_end;
extern "C" unsigned char _binary_bf16_fenced_o_start, _binary_bf16_fenced_o_end;
extern "C" unsigned char _binary_fp8_fenced_o_start, _binary_fp8_fenced_o_end;
static const char* names[] = {"gk_generated_feed_bf16", "gk_generated_feed_fp8", "gk_generated_feed_bf16_fenced", "gk_generated_feed_fp8_fenced"};
static int which(const char* s) { for(int i=0;i<4;++i) if(!std::strcmp(s,names[i])) return i; return -1; }
template<class T> static void map(T& p,int dim,int idx,float a,int hi) {
    p.mapping[dim].indexSpaceDim=idx; p.mapping[dim].a=a;
    p.mapping[dim].start_b=0; p.mapping[dim].end_b=hi;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t* n,GuidInfo* g) {
    *n=d==DEVICE_ID_GAUDI2?4:0;
    if(g) for(unsigned i=0;i<*n;++i) std::strcpy(g[i].name,names[i]); return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out) {
    int id=which(in->guid.name); if(id<0) return GLUE_NODE_NOT_FOUND;
    if(in->inputTensorNr!=(id>=2?2u:1u)||in->outputTensorNr!=1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    const auto& y=in->outputTensors[0].geometry; const auto& x=in->inputTensors[0].geometry;
    int width=(id%2)?256:128, n=y.maxSizes[0], k=y.maxSizes[1], e=y.maxSizes[2];
    if(n%width||k%32||e<1||x.maxSizes[0]!=n||x.maxSizes[1]!=e) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->kernel.paramsNr=0; out->indexSpaceRank=3;
    out->indexSpaceGeometry[0]=n/width; out->indexSpaceGeometry[1]=k/32; out->indexSpaceGeometry[2]=e;
    auto& s=out->inputTensorAccessPattern[0]; s.allRequired=0;
    map(s,0,0,width,width-1); map(s,1,2,1,0);
    if(id>=2) out->inputTensorAccessPattern[1].allRequired=1;
    auto& w=out->outputTensorAccessPattern[0]; w.allRequired=0;
    map(w,0,0,width,width-1); map(w,1,1,32,31); map(w,2,2,1,0);
    unsigned char* starts[]={&_binary_bf16_o_start,&_binary_fp8_o_start,&_binary_bf16_fenced_o_start,&_binary_fp8_fenced_o_start};
    unsigned char* ends[]={&_binary_bf16_o_end,&_binary_fp8_o_end,&_binary_bf16_fenced_o_end,&_binary_fp8_fenced_o_end};
    unsigned char* begin=starts[id];unsigned char* end=ends[id];
    unsigned cap=out->kernel.elfSize; out->kernel.elfSize=end-begin;
    if(cap<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize); return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams* in,ShapeInferenceOutput*) {
    return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* ls,uint32_t* n) {
    if(which(in->guid.name)<0) return GLUE_NODE_NOT_FOUND;
    unsigned cap=*n; *n=1; if(!ls||!cap) return GLUE_SUCCESS;
    std::memset(ls[0].inputs[0].layout,'x',sizeof(ls[0].inputs[0].layout));
    if(in->inputTensorNr==2) std::memset(ls[0].inputs[1].layout,'x',sizeof(ls[0].inputs[1].layout));
    std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout)); return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion(){return 1;}
