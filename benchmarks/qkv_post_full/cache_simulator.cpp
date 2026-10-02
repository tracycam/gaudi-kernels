// Local ISA-only probe. Uses installed simulator API; no vendor headers copied.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
template<class T> void read(const char*path,std::vector<T>&data){std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(T));if(!f)throw std::runtime_error("short input file");}
template<class T> void write(const char*path,const std::vector<T>&data){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));if(!f)throw std::runtime_error("output write failed");}
static Tensor tensor(TensorDataType type,const std::vector<unsigned>&sizes){
 Tensor t{};t.geometry.dims=sizes.size();t.geometry.dataType=type;t.quantizationParam.scale=1;
 for(unsigned d=0;d<MAX_TENSOR_DIM;++d)t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d<sizes.size()?sizes[d]:1;
 return t;
}
static tpc_tests::TensorDesc2 descriptor(void*data,const std::vector<unsigned>&sizes,unsigned log2){
 tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=reinterpret_cast<uint64_t>(data);d.paddingValue=0;
 d.configuration=log2|(((1u<<sizes.size())-1u)<<8)|((sizes.size()-1u)<<16);unsigned stride=1;
 for(unsigned dim=0;dim<5;++dim){unsigned size=dim<sizes.size()?sizes[dim]:1;d.dimDescriptors[dim].size=size;d.dimDescriptors[dim].stride=stride;stride*=size;}
 return d;
}
int main(int argc,char**argv){
 if(argc!=5){std::cerr<<"usage: simulator arithmetic M input_dir output_dir\n";return 2;}
 if(!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 3;
 std::string kind=argv[1],input=argv[3],output=argv[4];bool cached=kind=="cache"||kind=="cache-col";unsigned m=std::stoul(argv[2]);if(m!=3)return 4;
 std::vector<uint16_t>x(m*3392),cosine(m*64),sine(m*64),query(m*3072,0xcdcd),key(m*192,0xcdcd),value(m*128,0xcdcd);std::vector<uint8_t>elf(1<<20);
 read((input+"/qkv.bin").c_str(),x);read((input+"/cosine.bin").c_str(),cosine);read((input+"/sine.bin").c_str(),sine);
 std::vector<uint16_t>cache(32768*64,0x7fc1);std::vector<int32_t>positions={0,128,32767};
 for(unsigned row=0;row<m;++row){std::memcpy(cache.data()+positions[row]*64,cosine.data()+row*64,64);std::memcpy(cache.data()+positions[row]*64+32,sine.data()+row*64,64);}
 std::vector<unsigned>ps=kind=="cache-col"?std::vector<unsigned>{1,m}:std::vector<unsigned>{m};
 Tensor inputs[]={tensor(DATA_BF16,{3392,m}),cached?tensor(DATA_BF16,{64,32768}):tensor(DATA_BF16,{64,1,m}),cached?tensor(DATA_I32,ps):tensor(DATA_BF16,{64,1,m})};
 Tensor outputs[]={cached?tensor(DATA_BF16,{3072,m}):tensor(DATA_BF16,{192,16,m}),tensor(DATA_BF16,{192,1,m}),tensor(DATA_BF16,{128,1,m})};
 TensorAccessPattern input_ap[3]{},output_ap[3]{};AuxTensor aux[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=3;in.outputTensorNr=3;in.inputTensors=inputs;in.outputTensors=outputs;
 std::strcpy(in.guid.name,cached?"gk_qkv_post_cache_bf16_v2":("gk_qkv_post_"+kind+"_v1").c_str());float scale=.612f;in.nodeParams.nodeParams=&scale;in.nodeParams.nodeParamsSize=4;
 out.inputTensorAccessPattern=input_ap;out.outputTensorAccessPattern=output_ap;out.auxiliaryTensors=aux;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=InstantiateTpcKernel(&in,&out);if(status!=GLUE_SUCCESS){std::cerr<<"glue error "<<status<<'\n';return 5;}
 std::vector<tpc_tests::TensorDesc2>desc={descriptor(x.data(),{3392,m},1),cached?descriptor(cache.data(),{64,32768},1):descriptor(cosine.data(),{64,1,m},1),cached?descriptor(positions.data(),ps,2):descriptor(sine.data(),{64,1,m},1),cached?descriptor(query.data(),{3072,m},1):descriptor(query.data(),{192,16,m},1),descriptor(key.data(),{192,1,m},1),descriptor(value.data(),{128,1,m},1)};
 VPEStats stats;std::cout<<"TPC_RUNNER=0; local ISA simulation only"<<std::endl;
 unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);
 write((output+"/query.bin").c_str(),query);write((output+"/key.bin").c_str(),key);write((output+"/value.bin").c_str(),value);
 std::cout<<"{\"cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"device_verified\":false}"<<std::endl;
}
