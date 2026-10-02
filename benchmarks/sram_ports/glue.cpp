#include <tpc_kernel_lib_interface.h>
#include <cstring>
using namespace tpc_lib_api;
#ifndef PORT_WRITE_UNROLL
#define PORT_WRITE_UNROLL 8
#endif
extern "C" unsigned char _binary_read_o_start,_binary_read_o_end;
extern "C" unsigned char _binary_write_o_start,_binary_write_o_end;
extern "C" unsigned char _binary_copy_o_start,_binary_copy_o_end;
extern "C" unsigned char _binary_read_il_o_start,_binary_read_il_o_end;
extern "C" unsigned char _binary_write_il_o_start,_binary_write_il_o_end;
extern "C" unsigned char _binary_copy_il_o_start,_binary_copy_il_o_end;
extern "C" unsigned char _binary_drain_o_start,_binary_drain_o_end;
extern "C" unsigned char _binary_write_flat_o_start,_binary_write_flat_o_end;
static const char* names[]={"gk_sram_port_read","gk_sram_port_write","gk_sram_port_copy","gk_sram_port_read_il","gk_sram_port_write_il","gk_sram_port_copy_il","gk_sram_port_drain","gk_sram_port_write_flat"};
static int which(const char*s){for(int i=0;i<8;++i)if(!std::strcmp(s,names[i]))return i;return -1;}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t*n,GuidInfo*g){*n=d==DEVICE_ID_GAUDI2?8:0;if(g)for(unsigned i=0;i<*n;++i)std::strcpy(g[i].name,names[i]);return GLUE_SUCCESS;}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out){
 int kernel=which(in->guid.name);if(kernel<0)return GLUE_NODE_NOT_FOUND;bool drain=kernel==6,flat=kernel==7;int id=drain?2:kernel%3;bool il=kernel>=3&&kernel<6;int rd=il?2:1,td=il?1:2;
 if(in->inputTensorNr!=(drain?2u:1u)||in->outputTensorNr!=1)return GLUE_INCOMPATIBLE_INPUT_COUNT;
 if(in->nodeParams.nodeParamsSize!=8||!in->nodeParams.nodeParams)return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
 const auto&x=in->inputTensors[0].geometry;const auto&y=in->outputTensors[0].geometry;
 int rows=((int*)in->nodeParams.nodeParams)[0],reps=((int*)in->nodeParams.nodeParams)[1];
 if(flat&&(rows<=0||y.maxSizes[1]%rows))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 int tasks=flat?y.maxSizes[1]/rows:id==0?y.maxSizes[1]:y.maxSizes[td];
 if(rows<=0||rows%8||reps<=0||tasks<=0)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(id==1&&rows%PORT_WRITE_UNROLL)return GLUE_INCOMPATIBLE_INPUT_SIZE;
 if(id!=1&&(x.maxSizes[0]!=64||x.maxSizes[rd]!=unsigned(rows)||x.maxSizes[td]!=unsigned(tasks)))return GLUE_INCOMPATIBLE_INPUT_SIZE;
 out->kernel.paramsNr=2;std::memcpy(out->kernel.scalarParams,in->nodeParams.nodeParams,8);
 out->indexSpaceRank=1;out->indexSpaceGeometry[0]=tasks;
 auto set=[](auto&p,int dim,int idx,float a,int hi){p.mapping[dim].indexSpaceDim=idx;p.mapping[dim].a=a;p.mapping[dim].start_b=0;p.mapping[dim].end_b=hi;};
 auto&xp=out->inputTensorAccessPattern[0];auto&yp=out->outputTensorAccessPattern[0];
 if(drain)out->inputTensorAccessPattern[1].allRequired=1;
 if(id==1)xp.allRequired=1;
 else{set(xp,0,0,0,63);set(xp,rd,0,0,rows-1);set(xp,td,0,1,0);}
 set(yp,0,0,0,63);if(flat)set(yp,1,0,rows,rows-1);else if(id==0)set(yp,1,0,1,0);else{set(yp,rd,0,0,rows-1);set(yp,td,0,1,0);}
 unsigned char*starts[]={&_binary_read_o_start,&_binary_write_o_start,&_binary_copy_o_start,&_binary_read_il_o_start,&_binary_write_il_o_start,&_binary_copy_il_o_start,&_binary_drain_o_start,&_binary_write_flat_o_start};
 unsigned char*ends[]={&_binary_read_o_end,&_binary_write_o_end,&_binary_copy_o_end,&_binary_read_il_o_end,&_binary_write_il_o_end,&_binary_copy_il_o_end,&_binary_drain_o_end,&_binary_write_flat_o_end};
 unsigned cap=out->kernel.elfSize;out->kernel.elfSize=ends[kernel]-starts[kernel];
 if(cap<out->kernel.elfSize)return GLUE_INSUFFICIENT_ELF_BUFFER;
 std::memcpy(out->kernel.kernelElf,starts[kernel],out->kernel.elfSize);return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId,const ShapeInferenceParams*in,ShapeInferenceOutput*){return which(in->pGuid?in->pGuid->name:in->guid.name)<0?GLUE_NODE_NOT_FOUND:GLUE_SUCCESS;}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*ls,uint32_t*n){if(which(in->guid.name)<0)return GLUE_NODE_NOT_FOUND;unsigned cap=*n;*n=1;if(ls&&cap){for(unsigned i=0;i<in->inputTensorNr;++i)std::memset(ls[0].inputs[i].layout,'x',sizeof(ls[0].inputs[i].layout));std::memset(ls[0].outputs[0].layout,'x',sizeof(ls[0].outputs[0].layout));}return GLUE_SUCCESS;}
extern "C" uint64_t GetLibVersion(){return 1;}
