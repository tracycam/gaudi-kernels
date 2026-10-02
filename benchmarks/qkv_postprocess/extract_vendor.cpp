// CPU-only public perf-library instantiation. No Synapse/device APIs are linked.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(unsigned heads){
 Tensor t{};t.geometry.dims=3;t.geometry.dataType=DATA_BF16;t.quantizationParam.scale=1;
 for(unsigned i=0;i<MAX_TENSOR_DIM;++i){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?64:i==1?heads:1;t.permutation[i]=i;}
 std::memset(t.layout.layout,'x',sizeof(t.layout.layout));
 return t;
}
int main(int argc,char**argv){
 if(argc!=3){std::cerr<<"usage: extract_vendor libtpc_kernels.so output.elf\n";return 2;}
 void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<'\n';return 3;}
 auto guids=reinterpret_cast<pfnGetKernelGuids>(dlsym(lib,"GetKernelGuids"));
 auto instantiate=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));
 if(!guids||!instantiate)return 4;
 uint32_t count=0;auto status=guids(DEVICE_ID_GAUDI2,&count,nullptr);std::cerr<<"enumerate="<<status<<" count="<<count<<'\n';
 if(count==0||count>100000)return 5;
 std::vector<GuidInfo>list(count);status=guids(DEVICE_ID_GAUDI2,&count,list.data());if(status!=GLUE_SUCCESS)return 6;
 GuidInfo selected{};bool found=false;
 for(const auto&g:list)if(std::strstr(g.name,"rope_st2_fwd_bf16")){std::cerr<<"guid="<<g.name<<'\n';if(!std::strcmp(g.name,"rope_st2_fwd_bf16")){selected=g;found=true;}}
 if(!found)return 7;
 Tensor inputs[]={tensor(16),tensor(1),tensor(1)},outputs[]={tensor(16)};
 TensorAccessPattern inputAP[3]{},outputAP[1]{};AuxTensor aux[16]{};
 std::vector<unsigned char>elf(1<<20);uint32_t params[2]={0,0};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.apiVersion=1;in.deviceId=DEVICE_ID_GAUDI2;in.guid=selected;in.inputTensors=inputs;in.inputTensorNr=3;in.outputTensors=outputs;in.outputTensorNr=1;in.maxAvailableTpc=24;in.validInputTensors=7;in.validOutputTensors=1;
 in.nodeParams.nodeParams=params;in.nodeParams.nodeParamsSize=sizeof(params);
 out.inputTensorAccessPattern=inputAP;out.outputTensorAccessPattern=outputAP;out.auxiliaryTensors=aux;out.auxiliaryTensorNr=16;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 status=instantiate(&in,&out);std::cerr<<"instantiate="<<status<<" elf_size="<<out.kernel.elfSize<<" aux="<<out.auxiliaryTensorNr<<'\n';if(status!=GLUE_SUCCESS)return 8;
 std::cerr<<"kernel_buffer_replaced="<<(out.kernel.kernelElf!=elf.data())<<'\n';
 if(!out.kernel.kernelElf||std::memcmp(out.kernel.kernelElf,"\177ELF",4)){std::cerr<<"returned buffer is not ELF\n";return 9;}
 std::ofstream f(argv[2],std::ios::binary);f.write(reinterpret_cast<char*>(out.kernel.kernelElf),out.kernel.elfSize);if(!f)return 10;
 std::cout<<"{\"guid\":\""<<selected.name<<"\",\"index_space_rank\":"<<out.indexSpaceRank<<",\"geometry\":[";
 for(unsigned i=0;i<out.indexSpaceRank;++i)std::cout<<(i?",":"")<<out.indexSpaceGeometry[i];
 std::cout<<"],\"scalar_params\":[";for(unsigned i=0;i<out.kernel.paramsNr;++i)std::cout<<(i?",":"")<<out.kernel.scalarParams[i];
 std::cout<<"],\"auxiliary_count\":"<<out.auxiliaryTensorNr<<",\"device_acquired\":false}\n";
}
