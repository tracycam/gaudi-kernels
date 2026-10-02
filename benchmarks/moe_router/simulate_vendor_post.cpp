// Offline simulation of one extracted vendor primitive. No device APIs.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned n){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i?1:n;t.permutation[i]=i;}return t;}
tpc_tests::TensorDesc2 desc(void*p,unsigned n){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)p;d.configuration=2u|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){d.dimDescriptors[i].size=i?1:n;d.dimDescriptors[i].stride=stride;stride*=i?1:n;}return d;}
void read(const char*n,std::vector<uint32_t>&v){std::ifstream f(n,std::ios::binary);f.read((char*)v.data(),v.size()*4);if(!f)throw std::runtime_error("short input");}
int main(int argc,char**argv){
 if(argc<7||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 bool gather=std::strcmp(argv[2],"gather")==0,reduce=std::strcmp(argv[2],"reduce")==0;
 unsigned na=gather?384:8,nb=gather?8:1,ny=reduce?1:8;std::vector<uint32_t>a(na),b(nb),y(ny,0xdeadbeef);read(argv[3],a);if(!reduce)read(argv[4],b);
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f)return 3;std::vector<unsigned char>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
 std::vector<Tensor>inputs={tensor(DATA_F32,na)};if(!reduce)inputs.push_back(tensor(gather?DATA_I32:DATA_F32,nb));Tensor output=tensor(DATA_F32,ny);
 TensorAccessPattern ia[2]{},oa{};for(auto&v:ia)v.allRequired=1;oa.allRequired=1;
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=&output;std::strcpy(in.guid.name,"vendor_post_primitive_sim");
 out.indexSpaceRank=5;for(unsigned i=0;i<5;++i)out.indexSpaceGeometry[i]=1;out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=&oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 unsigned params=std::stoul(argv[6]);if(argc!=int(7+params))return 4;out.kernel.paramsNr=params;for(unsigned i=0;i<params;++i)out.kernel.scalarParams[i]=std::stoul(argv[7+i]);
 std::vector<tpc_tests::TensorDesc2>d={desc(a.data(),na)};if(!reduce)d.push_back(desc(b.data(),nb));d.push_back(desc(y.data(),ny));VPEStats stats;unsigned cycles=tpc_tests::RunSimulation(in,out,d,stats);
 std::ofstream o(argv[5],std::ios::binary);o.write((char*)y.data(),4*y.size());if(!o)return 5;
 std::cout<<"{\"TPC_RUNNER\":0,\"simulator_cycles_not_device_time\":"<<cycles<<",\"scalar_input0_loads\":"<<stats.scalarLoadPerTensor[0]<<",\"scalar_input1_loads\":"<<stats.scalarLoadPerTensor[1]<<",\"vector_input0_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"vector_input1_loads\":"<<stats.vectorLoadPerTensor[1]<<"}\n";
}
