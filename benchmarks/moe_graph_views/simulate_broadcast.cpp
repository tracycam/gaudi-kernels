#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(unsigned width,unsigned rows){Tensor t{};t.geometry.dims=2;t.geometry.dataType=DATA_BF16;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?width:i==1?rows:1;return t;}
tpc_tests::TensorDesc2 desc(void*data,unsigned width,unsigned rows){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=1u|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned n=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=n;d.dimDescriptors[i].stride=stride;stride*=n;}return d;}
int main(int argc,char**argv){
 if(argc!=3||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 constexpr unsigned k=129,groups=(k+31)/32,rows=groups*32;
 std::vector<uint16_t>x(k),y(rows*128,0xdead);for(unsigned i=0;i<k;++i)x[i]=uint16_t(i*499+17);
 x[0]=0;x[1]=0x8000;x[2]=1;x[3]=0x8001;x[4]=0x7f7f;x[5]=0xff7f;
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);std::vector<unsigned char>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
 Tensor input=tensor(k,1),output=tensor(128,rows);TensorAccessPattern ia{},oa{};ia.allRequired=oa.allRequired=1;
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=in.outputTensorNr=1;in.inputTensors=&input;in.outputTensors=&output;std::strcpy(in.guid.name,"broadcast_probe");
 out.indexSpaceRank=1;out.indexSpaceGeometry[0]=groups;out.inputTensorAccessPattern=&ia;out.outputTensorAccessPattern=&oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 std::vector<tpc_tests::TensorDesc2>d={desc(x.data(),k,1),desc(y.data(),128,rows)};VPEStats stats;
 unsigned cycles=tpc_tests::RunSimulation(in,out,d,stats),mismatch=0;
 for(unsigned row=0;row<rows;++row)for(unsigned lane=0;lane<128;++lane)if(y[row*128+lane]!=(row<k?x[row]:0))++mismatch;
 std::ofstream actual(argv[2],std::ios::binary);actual.write((char*)y.data(),y.size()*2);
 std::cout<<"{\"mismatch\":"<<mismatch<<",\"checked_words\":"<<y.size()<<",\"simulator_cycles\":"<<cycles<<",\"TPC_RUNNER\":0}\n";
 return mismatch?1:0;
}
