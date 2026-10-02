// Full old/new down simulation, distinct activations per route and repeated IDs.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned width,unsigned rows){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?width:i==1?rows:1;return t;}
tpc_tests::TensorDesc2 desc(void*data,unsigned width,unsigned rows,unsigned log2){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned n=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=n;d.dimDescriptors[i].stride=stride;stride*=n;}return d;}
template<class T>void write(const char*name,const std::vector<T>&v){std::ofstream f(name,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));}
int main(int argc,char**argv){
 if(argc!=3||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 constexpr unsigned experts=2,tokens=2,routes=8,k=256,n=6144,slots=tokens*routes,tasks=slots*12;
 std::vector<uint8_t>w(experts*n*k/2),s(experts*n*k/32);std::vector<uint16_t>x(slots*k),table(512);std::vector<int32_t>ids(slots);std::vector<float>y(slots*n,-99);
 for(unsigned i=0;i<w.size();++i)w[i]=(i*47+i/257)%256;
 for(unsigned i=0;i<s.size();++i)s[i]=122+(i*7+i/13)%8;
 for(unsigned i=0;i<x.size();++i)x[i]=(0x3c00+(i*37)%1024)^(i%3==0?0x8000:0);
 for(unsigned i=0;i<slots;++i)ids[i]=(i+i/routes)%experts;
 const uint16_t values[]={0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0,0x8000,0xbf00,0xbf80,0xbfc0,0xc000,0xc040,0xc080,0xc0c0};
 for(unsigned i=0;i<256;++i){table[2*i]=values[i&15];table[2*i+1]=values[i>>4];}
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);std::vector<uint8_t>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
 std::vector<Tensor>inputs={tensor(DATA_U8,256,experts*12*k),tensor(DATA_U8,512,experts*12*k/32),tensor(DATA_BF16,k,slots),tensor(DATA_BF16,512,1),tensor(DATA_I32,routes,tokens)};
 Tensor output=tensor(DATA_F32,slots*n,1);TensorAccessPattern ia[5]{},oa{};for(auto&a:ia)a.allRequired=1;oa.allRequired=1;
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=5;in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=&output;std::strcpy(in.guid.name,"full_down_simulator");
 out.indexSpaceRank=1;out.indexSpaceGeometry[0]=tasks;out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=&oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();out.kernel.paramsNr=1;out.kernel.scalarParams[0]=(uint64_t(1)<<32)/routes;
 std::vector<tpc_tests::TensorDesc2>d={desc(w.data(),256,experts*12*k,0),desc(s.data(),512,experts*12*k/32,0),desc(x.data(),k,slots,1),desc(table.data(),512,1,1),desc(ids.data(),routes,tokens,2),desc(y.data(),slots*n,1,2)};VPEStats stats;
 unsigned cycles=tpc_tests::RunSimulation(in,out,d,stats);
 write(argv[2],y);write("weights.bin",w);write("scales.bin",s);write("activation.bin",x);write("ids.bin",ids);write("table.bin",table);
 std::cout<<"{\"output_words\":"<<y.size()<<",\"tasks\":"<<tasks<<",\"simulator_cycles\":"<<cycles<<",\"weight_tensor_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_tensor_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"activation_tensor_loads\":"<<stats.vectorLoadPerTensor[2]<<",\"TPC_RUNNER\":0}\n";
}
