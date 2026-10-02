// ISA-only actual shared library/glue gate. Never acquire a device.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,std::vector<unsigned>shape){Tensor t{};t.geometry.dims=shape.size();t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i<shape.size()?shape[i]:1;return t;}
tpc_tests::TensorDesc2 desc(void*data,std::vector<unsigned>shape,unsigned log2){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=log2|(((1u<<shape.size())-1)<<8)|((shape.size()-1)<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i<shape.size()?shape[i]:1;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}return d;}
template<class T>void read(const char*p,std::vector<T>&v){std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("short input");}
int main(int argc,char**argv){
 if(argc!=7||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 unsigned n=std::stoul(argv[2]),g=std::stoul(argv[3]);if(!n||n>513||!g||g>32)return 3;
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());auto fn=(GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*))dlsym(h,"InstantiateTpcKernel");if(!fn)return 4;
 unsigned blocks=(n+127)/128;std::vector<uint8_t>w(g*n*128),elf(1<<20);std::vector<float>s(blocks*g);std::vector<uint16_t>y(n*g*128+128,0xcdcd);read(argv[4],w);read(argv[5],s);
 Tensor input[]={tensor(DATA_F8_143,{128,n,g}),tensor(DATA_F32,{g,blocks})},output[]={tensor(DATA_BF16,{g*128,n})};TensorAccessPattern ia[2]{},oa[1]{};
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=2;in.outputTensorNr=1;in.inputTensors=input;in.outputTensors=output;std::strcpy(in.guid.name,"gk_block128_decode_v1");out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();if(fn(&in,&out)!=GLUE_SUCCESS)return 5;
 std::vector<tpc_tests::TensorDesc2>d={desc(w.data(),{128,n,g},0),desc(s.data(),{g,blocks},2),desc(y.data(),{g*128,n},1)};
 VPEStats stats;auto cycles=tpc_tests::RunSimulation(in,out,d,stats);
 for(unsigned i=n*g*128;i<y.size();++i)if(y[i]!=0xcdcd)throw std::runtime_error("guard overwritten");
 std::ofstream f(argv[6],std::ios::binary);f.write((char*)y.data(),n*g*128*2);if(!f)return 6;
 std::cout<<"{\"TPC_RUNNER\":0,\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"guard_intact\":true,\"device_used\":false}\n";dlclose(h);
}
