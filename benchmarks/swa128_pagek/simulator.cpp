#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
template<class T>void read(std::string p,std::vector<T>&v){std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error(p);}
Tensor tensor(TensorDataType type,std::vector<unsigned>s){Tensor t{};t.geometry.dims=s.size();t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned d=0;d<MAX_TENSOR_DIM;++d){t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d<s.size()?s[d]:1;t.permutation[d]=d;}return t;}
tpc_tests::TensorDesc2 desc(void*p,std::vector<unsigned>s,unsigned log2){tpc_tests::TensorDesc2 t{};t.baseAddrUnion.baseAddr=(uint64_t)p;t.configuration=log2|(((1u<<s.size())-1)<<8)|((s.size()-1)<<16);unsigned stride=1;for(unsigned d=0;d<5;++d){unsigned n=d<s.size()?s[d]:1;t.dimDescriptors[d].size=n;t.dimDescriptors[d].stride=stride;stride*=n;}return t;}
int main(int argc,char**argv){
 if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);std::vector<unsigned char>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());if(!f||std::memcmp(elf.data(),"\177ELF",4))return 3;
 std::string input=argv[2];std::vector<uint16_t>q(2*192),k(3*192*128);std::vector<int32_t>m(3);std::vector<float>y(2*128,777);read(input+"/q.bin",q);read(input+"/kt.bin",k);read(input+"/metadata.bin",m);
 Tensor ins[]={tensor(DATA_BF16,{192,2}),tensor(DATA_BF16,{128,192,3}),tensor(DATA_I32,{3})},outs[]={tensor(DATA_F32,{128,2})};TensorAccessPattern ia[3]{},oa[1]{};AuxTensor aux[1]{};
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;std::strcpy(in.guid.name,"gk_swa128_pagek_qk_offline_v0");in.inputTensorNr=3;in.outputTensorNr=1;in.inputTensors=ins;in.outputTensors=outs;
 out.indexSpaceRank=1;for(unsigned d=0;d<MAX_INDEX_SPACE_DIM_SIZE;++d)out.indexSpaceGeometry[d]=d?1:2;out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.auxiliaryTensors=aux;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 std::vector<tpc_tests::TensorDesc2>ds={desc(q.data(),{192,2},1),desc(k.data(),{128,192,3},1),desc(m.data(),{3},2),desc(y.data(),{128,2},2)};VPEStats stats;auto cycles=tpc_tests::RunSimulation(in,out,ds,stats);
 std::ofstream output(argv[3],std::ios::binary);output.write((char*)y.data(),y.size()*4);std::cout<<"{\"cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"device_verified\":false}"<<std::endl;
}
