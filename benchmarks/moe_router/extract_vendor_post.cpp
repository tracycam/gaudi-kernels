// CPU-only public-library instantiation of the actual post-TopK GUIDs.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned width){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<MAX_TENSOR_DIM;++i){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i?1:width;t.permutation[i]=i;}std::memset(t.layout.layout,'x',sizeof(t.layout.layout));return t;}
int main(int argc,char**argv){
 if(argc!=4)return 2;void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<'\n';return 3;}
 auto get=reinterpret_cast<pfnGetKernelGuids>(dlsym(lib,"GetKernelGuids"));auto instantiate=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));if(!get||!instantiate)return 4;
 uint32_t count=0;if(get(DEVICE_ID_GAUDI2,&count,nullptr)!=GLUE_SUCCESS||count>100000)return 5;std::vector<GuidInfo>gs(count);if(get(DEVICE_ID_GAUDI2,&count,gs.data())!=GLUE_SUCCESS)return 6;
 GuidInfo selected{};bool found=false;for(auto&g:gs)if(!std::strcmp(g.name,argv[2])){selected=g;found=true;}if(!found)return 7;
 bool gather=std::strstr(argv[2],"gather")!=nullptr,reduce=std::strstr(argv[2],"reduce")!=nullptr;
 std::vector<Tensor>inputs={tensor(DATA_F32,gather?384:8)};if(!reduce)inputs.push_back(tensor(gather?DATA_I32:DATA_F32,gather?8:1));
 Tensor output=tensor(DATA_F32,reduce?1:8);TensorAccessPattern ia[2]{},oa[1]{};AuxTensor aux[16]{};std::vector<unsigned char>elf(1<<20);unsigned axis=0;
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.apiVersion=1;in.deviceId=DEVICE_ID_GAUDI2;in.guid=selected;in.inputTensors=inputs.data();in.inputTensorNr=inputs.size();in.outputTensors=&output;in.outputTensorNr=1;in.maxAvailableTpc=24;in.validInputTensors=(1u<<inputs.size())-1;in.validOutputTensors=1;
 if(gather||reduce){in.nodeParams.nodeParams=&axis;in.nodeParams.nodeParamsSize=4;}
 out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.auxiliaryTensors=aux;out.auxiliaryTensorNr=16;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=instantiate(&in,&out);std::cerr<<"status="<<status<<" elf_size="<<out.kernel.elfSize<<" aux="<<out.auxiliaryTensorNr<<'\n';if(status!=GLUE_SUCCESS)return 8;
 if(out.auxiliaryTensorNr||!out.kernel.kernelElf||std::memcmp(out.kernel.kernelElf,"\177ELF",4))return 9;
 std::ofstream f(std::string(argv[3])+".o",std::ios::binary);f.write((char*)out.kernel.kernelElf,out.kernel.elfSize);if(!f)return 10;
 std::ofstream j(std::string(argv[3])+".json");j<<"{\"guid\":\""<<selected.name<<"\",\"device_acquired\":false,\"kernel_buffer_replaced\":"<<(out.kernel.kernelElf!=elf.data()?"true":"false")<<",\"index_space_rank\":"<<out.indexSpaceRank<<",\"geometry\":[";
 for(unsigned i=0;i<out.indexSpaceRank;++i)j<<(i?",":"")<<out.indexSpaceGeometry[i];j<<"],\"scalar_params\":[";
 for(unsigned i=0;i<out.kernel.paramsNr;++i)j<<(i?",":"")<<out.kernel.scalarParams[i];j<<"]}\n";return j?0:11;
}
