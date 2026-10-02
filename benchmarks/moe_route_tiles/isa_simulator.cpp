// Offline actual-ELF execution only. The old gate ELF is supplied from sealed assets.
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
struct Buffer{std::string path;std::vector<unsigned>shape;TensorDataType type;unsigned log2;std::vector<unsigned char>bytes;};
int main(int argc,char**argv){
 if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::ifstream spec(argv[2]);std::string guid;unsigned ni,no,np;spec>>guid>>ni>>no>>np;std::vector<int>params(np);for(int&v:params)spec>>v;
 std::vector<Buffer>buf(ni+no);std::vector<Tensor>in(ni),out(no);std::vector<tpc_tests::TensorDesc2>ds;
 for(unsigned i=0;i<buf.size();++i){auto&b=buf[i];std::string type;unsigned rank;spec>>type>>rank;b.type=type=="bf16"?DATA_BF16:type=="f32"?DATA_F32:DATA_I32;b.log2=type=="bf16"?1:2;b.shape.resize(rank);unsigned n=1;for(auto&v:b.shape){spec>>v;n*=v;}spec>>b.path;b.bytes.resize(n*(1<<b.log2),0x5a);if(i<ni){std::ifstream f(b.path,std::ios::binary);f.read((char*)b.bytes.data(),b.bytes.size());if(!f)throw std::runtime_error("input read");}
  auto&t=i<ni?in[i]:out[i-ni];t.geometry.dims=rank;t.geometry.dataType=b.type;t.quantizationParam.scale=1;
  tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)b.bytes.data();d.configuration=b.log2|(((1u<<rank)-1)<<8)|((rank-1)<<16);unsigned stride=1;for(int j=0;j<5;++j){unsigned size=j<int(rank)?b.shape[j]:1;t.geometry.minSizes[j]=t.geometry.maxSizes[j]=size;d.dimDescriptors[j].size=size;d.dimDescriptors[j].stride=stride;stride*=size;}ds.push_back(d);
 }
 if(!spec)throw std::runtime_error("invalid descriptor specification");
 HabanaKernelParams p{};HabanaKernelInstantiation k{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=ni;p.outputTensorNr=no;p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,guid.c_str());p.nodeParams.nodeParams=params.data();p.nodeParams.nodeParamsSize=np*4;
 std::vector<TensorAccessPattern>ia(ni),oa(no);k.inputTensorAccessPattern=ia.data();k.outputTensorAccessPattern=oa.data();std::vector<unsigned char>elf(4<<20);k.kernel.kernelElf=elf.data();k.kernel.elfSize=elf.size();
 if(std::string(argv[3])!="-"){
  // The unchanged historical gate uses the same [W/128,C,batch] index space.
  std::ifstream f(argv[3],std::ios::binary);f.seekg(0,std::ios::end);auto size=f.tellg();f.seekg(0);if(size<=0||size>int(elf.size()))throw std::runtime_error("ELF size");f.read((char*)elf.data(),size);if(!f)throw std::runtime_error("ELF read");k.kernel.elfSize=size;k.kernel.paramsNr=np;std::memcpy(k.kernel.scalarParams,params.data(),np*4);k.indexSpaceRank=3;k.indexSpaceGeometry[0]=out[0].geometry.maxSizes[0]/128;k.indexSpaceGeometry[1]=out[0].geometry.maxSizes[1];k.indexSpaceGeometry[2]=out[0].geometry.maxSizes[2];
 }else{void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());auto fn=(Instantiate)dlsym(h,"InstantiateTpcKernel");if(!fn)throw std::runtime_error("missing glue");auto code=fn(&p,&k);if(code!=GLUE_SUCCESS)throw std::runtime_error("glue status "+std::to_string(code));}
 VPEStats stats;auto cycles=tpc_tests::RunSimulation(p,k,ds,stats);
 for(unsigned i=ni;i<buf.size();++i){std::ofstream f(buf[i].path,std::ios::binary);f.write((char*)buf[i].bytes.data(),buf[i].bytes.size());if(!f)throw std::runtime_error("output write");}
 std::cout<<"{\"status\":\"EXECUTED_ISA_ONLY\",\"cycles_not_device_time\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<"}"<<std::endl;
}
