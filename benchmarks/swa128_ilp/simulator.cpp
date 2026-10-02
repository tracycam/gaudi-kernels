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
 if(argc!=6){std::cerr<<"usage: simulator heads slots input_dir output_file elf\n";return 2;}
 if(!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 3;
 unsigned heads=std::stoul(argv[1]);if(heads<1||heads>16)return 4;unsigned slots=std::stoul(argv[2]);if(!slots||slots%128)return 4;std::string input=argv[3],output=argv[4];
 std::vector<uint16_t>q(heads*192),k(slots*192),v(slots*128),sink(heads),y(heads*128,0xcdcd);std::vector<int32_t>pages(2),starts(2),position(1);std::ifstream ef(argv[5],std::ios::binary|std::ios::ate);if(!ef)return 5;std::vector<uint8_t>elf(ef.tellg());ef.seekg(0);ef.read((char*)elf.data(),elf.size());if(!ef||elf.size()<4||std::memcmp(elf.data(),"\177ELF",4))return 5;
 read((input+"/query.bin").c_str(),q);read((input+"/key.bin").c_str(),k);read((input+"/value.bin").c_str(),v);read((input+"/sinks.bin").c_str(),sink);read((input+"/pages.bin").c_str(),pages);read((input+"/groups.bin").c_str(),starts);read((input+"/position.bin").c_str(),position);
 std::vector<unsigned> qshape={heads*192,1,1};
 Tensor inputs[]={tensor(DATA_BF16,qshape),tensor(DATA_BF16,{192,1,slots}),tensor(DATA_BF16,{128,1,slots}),tensor(DATA_I32,{2}),tensor(DATA_I32,{2}),tensor(DATA_I32,{1}),tensor(DATA_BF16,{heads})};
 Tensor outputs[]={tensor(DATA_BF16,{128,heads,1})};TensorAccessPattern input_ap[7]{},output_ap[1]{};AuxTensor aux[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=7;in.outputTensorNr=1;in.inputTensors=inputs;in.outputTensors=outputs;std::strcpy(in.guid.name,"gk_swa128_window_head_fast_fp32_offline");float scale=0.07216878364870322f;in.nodeParams.nodeParams=&scale;in.nodeParams.nodeParamsSize=4;
 out.inputTensorAccessPattern=input_ap;out.outputTensorAccessPattern=output_ap;out.auxiliaryTensors=aux;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 out.indexSpaceRank=1;for(unsigned d=0;d<MAX_INDEX_SPACE_DIM_SIZE;++d)out.indexSpaceGeometry[d]=d?1:heads;
 out.kernel.paramsNr=1;std::memcpy(out.kernel.scalarParams,&scale,4);
 std::vector<tpc_tests::TensorDesc2>desc={descriptor(q.data(),qshape,1),descriptor(k.data(),{192,1,slots},1),descriptor(v.data(),{128,1,slots},1),descriptor(pages.data(),{2},2),descriptor(starts.data(),{2},2),descriptor(position.data(),{1},2),descriptor(sink.data(),{heads},1),descriptor(y.data(),{128,heads,1},1)};
 VPEStats stats;std::cout<<"TPC_RUNNER=0; local ISA simulation only"<<std::endl;
 unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);write(output.c_str(),y);
 std::cout<<"{\"cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"device_verified\":false}"<<std::endl;
}
