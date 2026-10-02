// Audit-only glue: exact production ASM .text plus both real bucket decoders.
#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#define ELFS(X) X(layout_native) X(layout_historical) X(layout_gp) X(layout_down)
#define DECL(n) extern "C" unsigned char _binary_##n##_o_start,_binary_##n##_o_end;
ELFS(DECL)
static const char*names[]={"layout_native","layout_historical","layout_gp","layout_down"};
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int id=-1;for(int i=0;i<4;++i)if(!std::strcmp(in->guid.name,names[i]))id=i;if(id<0)return GLUE_NODE_NOT_FOUND;
 if(in->inputTensorNr!=(id<2?4:5)||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 for(unsigned i=0;i<in->inputTensorNr;++i)out->inputTensorAccessPattern[i].allRequired=1;
 out->outputTensorAccessPattern[0].allRequired=1;auto&y=in->outputTensors[0].geometry;
 if(id<2){out->kernel.paramsNr=3;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,12);out->indexSpaceRank=3;out->indexSpaceGeometry[0]=y.maxSizes[1]/32;out->indexSpaceGeometry[1]=y.maxSizes[0]/512;out->indexSpaceGeometry[2]=y.maxSizes[2];}
 else {out->kernel.paramsNr=1;out->kernel.scalarParams[0]=uint32_t((uint64_t(1)<<32)/in->inputTensors[4].geometry.maxSizes[0]);out->indexSpaceRank=1;out->indexSpaceGeometry[0]=y.maxSizes[0]/512;}
#define BEGIN(n) &_binary_##n##_o_start,
#define END(n) &_binary_##n##_o_end,
 unsigned char*begin[]={ELFS(BEGIN)},*end[]={ELFS(END)};unsigned cap=out->kernel.elfSize,size=end[id]-begin[id];out->kernel.elfSize=size;if(cap<size)return GLUE_INSUFFICIENT_ELF_BUFFER;std::memcpy(out->kernel.kernelElf,begin[id],size);return GLUE_SUCCESS;
}
