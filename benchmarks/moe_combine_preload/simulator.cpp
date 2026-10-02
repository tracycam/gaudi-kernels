// Actual embedded ELF and public glue only; TPC_RUNNER=0 forbids device use.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tpc_lib_api;
using Instantiate=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
struct Buffer{std::string path;std::vector<unsigned>shape;std::vector<unsigned char>bytes;};
int main(int argc,char**argv){
 if(argc!=3||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::ifstream spec(argv[2]);std::string guid;spec>>guid;
 std::vector<Buffer>buf(4);std::vector<Tensor>in(3),out(1);std::vector<tpc_tests::TensorDesc2>ds;
 for(unsigned i=0;i<4;++i){auto&b=buf[i];std::string type;unsigned rank;spec>>type>>rank;
  if(type!="u8"&&type!="f32")throw std::runtime_error("type");
  TensorDataType dtype=type=="u8"?DATA_U8:DATA_F32;unsigned log2=type=="u8"?0:2;
  if(rank!=2)throw std::runtime_error("rank");b.shape.resize(rank);unsigned n=1;
  for(auto&v:b.shape){spec>>v;n*=v;}spec>>b.path;b.bytes.resize(n*(1<<log2),0x5a);
  if(i<3){std::ifstream f(b.path,std::ios::binary);f.read((char*)b.bytes.data(),b.bytes.size());if(!f)throw std::runtime_error("input read");}
  auto&t=i<3?in[i]:out[0];t.geometry.dims=rank;t.geometry.dataType=dtype;t.quantizationParam.scale=1;
  tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)b.bytes.data();d.configuration=log2|(((1u<<rank)-1)<<8)|((rank-1)<<16);
  unsigned stride=1;for(int j=0;j<5;++j){unsigned size=j<int(rank)?b.shape[j]:1;t.geometry.minSizes[j]=t.geometry.maxSizes[j]=size;d.dimDescriptors[j].size=size;d.dimDescriptors[j].stride=stride;stride*=size;}ds.push_back(d);
 }
 if(!spec)throw std::runtime_error("spec");
 HabanaKernelParams p{};HabanaKernelInstantiation k{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=3;p.outputTensorNr=1;p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,guid.c_str());
 std::vector<TensorAccessPattern>ia(3),oa(1);k.inputTensorAccessPattern=ia.data();k.outputTensorAccessPattern=oa.data();std::vector<unsigned char>elf(4<<20);k.kernel.kernelElf=elf.data();k.kernel.elfSize=elf.size();
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());auto fn=(Instantiate)dlsym(h,"InstantiateTpcKernel");if(!fn)throw std::runtime_error("missing glue");
 auto code=fn(&p,&k);if(code!=GLUE_SUCCESS)throw std::runtime_error("glue status "+std::to_string(code));
 VPEStats stats;auto cycles=tpc_tests::RunSimulation(p,k,ds,stats);
 std::ofstream f(buf[3].path,std::ios::binary);f.write((char*)buf[3].bytes.data(),buf[3].bytes.size());
 if(!f)throw std::runtime_error("output write");
 std::cout<<"{\"status\":\"EXECUTED_ISA_ONLY\",\"cycles_not_device_time\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<"}"<<std::endl;
}
