// Execute the extracted installed-vendor ELF locally with real head stride192.
// RunSimulation is required to remain explicitly simulator-only (TPC_RUNNER=0).
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
template<class T>void read(const std::string&p,std::vector<T>&v){std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));if(!f)throw std::runtime_error("input read failed: "+p);}
template<class T>void write(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));}
Tensor tensor(unsigned heads){Tensor t{};t.geometry.dims=3;t.geometry.dataType=DATA_BF16;t.quantizationParam.scale=1;for(unsigned i=0;i<MAX_TENSOR_DIM;++i){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?64:i==1?heads:1;t.permutation[i]=i;}return t;}
tpc_tests::TensorDesc2 desc(void*p,unsigned heads,unsigned stride){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=reinterpret_cast<uint64_t>(p);d.configuration=1|(7<<8)|(2<<16);for(unsigned i=0;i<5;++i){d.dimDescriptors[i].size=i==0?64:i==1?heads:1;d.dimDescriptors[i].stride=i==0?1:i==1?stride:stride*heads;}return d;}
int main(int argc,char**argv){
 if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::ifstream ef(argv[1],std::ios::binary|std::ios::ate);std::vector<unsigned char>elf(ef.tellg());ef.seekg(0);ef.read(reinterpret_cast<char*>(elf.data()),elf.size());if(!ef||std::memcmp(elf.data(),"\177ELF",4))return 3;
 std::string input=argv[2],output=argv[3];std::vector<uint16_t>x(3*3392),c(3*64),s(3*64),q(3*3072),k(3*192);read(input+"/qkv.bin",x);read(input+"/cosine.bin",c);read(input+"/sine.bin",s);
 for(unsigned row=0;row<3;++row)for(unsigned key=0;key<2;++key){
  unsigned heads=key?1:16;std::vector<uint16_t>a(heads*192),y;std::memcpy(a.data(),x.data()+row*3392+(key?3072:0),a.size()*2);y=a;
  Tensor ins[]={tensor(heads),tensor(1),tensor(1)},outs[]={tensor(heads)};TensorAccessPattern ia[3]{},oa[1]{};AuxTensor aux[1]{};
  HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;std::strcpy(in.guid.name,"rope_st2_fwd_bf16");in.inputTensorNr=3;in.outputTensorNr=1;in.inputTensors=ins;in.outputTensors=outs;
  out.indexSpaceRank=5;for(unsigned i=0;i<5;++i)out.indexSpaceGeometry[i]=i==1?heads:1;
  out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.auxiliaryTensors=aux;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();out.kernel.paramsNr=5;
  // Installed public instantiation for Q: indexSpace [1,16,1,1,1], these exact
  // broadcast flags. K uses the same flags with the head coordinate bounded1.
  unsigned params[5]={1,1,1,1,0};std::memcpy(out.kernel.scalarParams,params,sizeof(params));
  std::vector<tpc_tests::TensorDesc2>ds={desc(a.data(),heads,192),desc(s.data()+row*64,1,64),desc(c.data()+row*64,1,64),desc(y.data(),heads,192)};
  VPEStats stats;unsigned cycles=tpc_tests::RunSimulation(in,out,ds,stats);
  std::memcpy((key?k.data()+row*192:q.data()+row*3072),y.data(),y.size()*2);
  std::cout<<"row="<<row<<" heads="<<heads<<" cycles="<<cycles<<" instructions="<<stats.instructionsExecuted<<'\n';
 }
 write(output+"/query.bin",q);write(output+"/key.bin",k);
}
