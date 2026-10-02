// Offline ISA only. No vendor simulator headers/implementation are copied.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
template<class T>void read(const char* name,std::vector<T>& data){std::ifstream f(name,std::ios::binary);f.read((char*)data.data(),data.size()*sizeof(T));if(!f)throw std::runtime_error("short file");}
template<class T>void write(const char* name,const std::vector<T>& data){std::ofstream f(name,std::ios::binary);f.write((char*)data.data(),data.size()*sizeof(T));if(!f)throw std::runtime_error("write failed");}
Tensor tensor(TensorDataType type,unsigned width){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;for(int i=0;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i?1:width;t.quantizationParam.scale=1;return t;}
tpc_tests::TensorDesc2 desc(void* data,unsigned width){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=2u|(3u<<8)|(1u<<16);unsigned stride=1;for(int i=0;i<5;++i){d.dimDescriptors[i].size=i?1:width;d.dimDescriptors[i].stride=stride;stride*=i?1:width;}return d;}
int main(int argc,char** argv){
 if(argc!=9||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0")){std::cerr<<"TPC_RUNNER=0 simulator elf scores ids weights output_ids sum factor renorm\n";return 2;}
 std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f)return 3;std::vector<unsigned char> elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
 std::vector<float> scores(384),weights(8,-99),sum(1,-99);std::vector<int> ids(8),out_ids(8,-99);read(argv[2],scores);read(argv[3],ids);
 for(auto id:ids)if(id<0||id>=384)return 4;
 Tensor inputs[]={tensor(DATA_F32,384),tensor(DATA_I32,8)},outputs[]={tensor(DATA_F32,8),tensor(DATA_I32,8),tensor(DATA_F32,1)};
 TensorAccessPattern in_ap[2]{},out_ap[3]{};for(auto& a:in_ap)a.allRequired=1;for(auto& a:out_ap)a.allRequired=1;
 struct Params{float factor;int renorm;int apply_scale;} params{std::stof(argv[7]),std::stoi(argv[8]),std::stod(argv[7])!=1.0};
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=2;in.outputTensorNr=3;in.inputTensors=inputs;in.outputTensors=outputs;
 std::strcpy(in.guid.name,"experimental_router_post_top8");in.nodeParams.nodeParams=&params;in.nodeParams.nodeParamsSize=sizeof(params);
 out.indexSpaceRank=1;out.indexSpaceGeometry[0]=1;out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;
 out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();out.kernel.paramsNr=3;std::memcpy(out.kernel.scalarParams,&params,sizeof(params));
 std::vector<tpc_tests::TensorDesc2> tensors={desc(scores.data(),384),desc(ids.data(),8),desc(weights.data(),8),desc(out_ids.data(),8),desc(sum.data(),1)};
 VPEStats stats;unsigned cycles=tpc_tests::RunSimulation(in,out,tensors,stats);
 write(argv[4],weights);write(argv[5],out_ids);write(argv[6],sum);
 std::cout<<"{\"TPC_RUNNER\":0,\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"score_scalar_loads\":"<<stats.scalarLoadPerTensor[0]<<",\"id_scalar_loads\":"<<stats.scalarLoadPerTensor[1]<<",\"score_vector_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"id_vector_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"device_runtime_verified\":false}\n";
}
