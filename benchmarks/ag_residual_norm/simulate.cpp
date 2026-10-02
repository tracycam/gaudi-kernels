// Strict offline native-ELF gate. This executable links no Synapse/device API.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned h,unsigned m,unsigned dims=2){Tensor t{};t.geometry.dims=dims;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?h:i==1?m:1;t.permutation[i]=i;}return t;}
tpc_tests::TensorDesc2 desc(void*p,unsigned h,unsigned m,unsigned log2){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)p;d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned n=i==0?h:i==1?m:1;d.dimDescriptors[i].size=n;d.dimDescriptors[i].stride=stride;stride*=n;}return d;}
template<class T>void read(const std::string&n,std::vector<T>&v){std::ifstream f(n,std::ios::binary);f.read((char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("input read: "+n);}
template<class T>void write(const std::string&n,std::vector<T>&v){std::ofstream f(n,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("output write: "+n);}
int main(int argc,char**argv){
 if(argc!=5||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 unsigned m=std::stoul(argv[2]),h=std::stoul(argv[3]);if(!m||m>8||!h||h>8192)return 3;std::string prefix=argv[4];
 std::vector<float>x(8*m*h);std::vector<uint16_t>r(m*h),w(h),ro(m*h,0x7fc0),y(m*h,0x7fc0);read("gathered.bin",x);read("residual.bin",r);read("gamma.bin",w);
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f)return 4;std::vector<unsigned char>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
 Tensor inputs[]={tensor(DATA_F32,h,8*m),tensor(DATA_BF16,h,m),tensor(DATA_BF16,h,1,1)},outputs[]={tensor(DATA_BF16,h,m),tensor(DATA_BF16,h,m)};
 TensorAccessPattern ia[3]{},oa[2]{};for(auto&a:ia)a.allRequired=1;for(auto&a:oa)a.allRequired=1;
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=3;in.outputTensorNr=2;in.inputTensors=inputs;in.outputTensors=outputs;std::strcpy(in.guid.name,"offline_ag8_norm");
 out.indexSpaceRank=1;out.indexSpaceGeometry[0]=m;out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();out.kernel.paramsNr=1;float eps=1e-6f;std::memcpy(out.kernel.scalarParams,&eps,4);
 std::vector<tpc_tests::TensorDesc2>d={desc(x.data(),h,8*m,2),desc(r.data(),h,m,1),desc(w.data(),h,1,1),desc(ro.data(),h,m,1),desc(y.data(),h,m,1)};VPEStats stats;
 unsigned cycles=tpc_tests::RunSimulation(in,out,d,stats);write(prefix+"-residual.bin",ro);write(prefix+"-norm.bin",y);
 std::cout<<"{\"M\":"<<m<<",\"H\":"<<h<<",\"simulator_cycles_not_device_time\":"<<cycles<<",\"gathered_vector_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"residual_vector_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"gamma_vector_loads\":"<<stats.vectorLoadPerTensor[2]<<",\"TPC_RUNNER\":0}\n";
}
