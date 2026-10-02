// Offline ISA probe. Do not copy proprietary simulator implementation/headers.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
template<class T> void read(const char* path,std::vector<T>& data){
 std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(T));
 if(!f)throw std::runtime_error("short input file");
}
template<class T> void write(const char* path,const std::vector<T>& data){
 std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
 if(!f)throw std::runtime_error("output write failed");
}
static Tensor tensor(TensorDataType dtype,unsigned width,unsigned rows=1){
 Tensor t{};t.geometry.dims=2;t.geometry.dataType=dtype;
 for(unsigned d=0;d<MAX_TENSOR_DIM;++d)t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d==0?width:(d==1?rows:1);
 t.quantizationParam.scale=1;t.quantizationParam.fp8bias=7;return t;
}
static tpc_tests::TensorDesc2 descriptor(void* data,unsigned width,unsigned element_log2,unsigned rows=1){
 tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=reinterpret_cast<uint64_t>(data);
 d.paddingValue=0;d.configuration=element_log2|(3u<<8)|(1u<<16);
 unsigned stride=1;
 for(unsigned dim=0;dim<5;++dim){unsigned size=dim==0?width:(dim==1?rows:1);d.dimDescriptors[dim].size=size;d.dimDescriptors[dim].stride=stride;stride*=size;}
 return d;
}
int main(int argc,char** argv){
 if(argc!=9){std::cerr<<"usage: simulator K M input.bin table.bin q.bin scale.bin variant baseline-elf\n";return 2;}
 const char* mode=std::getenv("TPC_RUNNER");
 if(!mode||std::strcmp(mode,"0")){std::cerr<<"TPC_RUNNER must explicitly be 0\n";return 3;}
 unsigned k=std::stoul(argv[1]),m=std::stoul(argv[2]);if((k!=129&&k!=513)||(m!=1&&m!=128)||(m==128&&k!=129))return 4;
 const bool baseline=std::strcmp(argv[7],"baseline")==0;
 const bool lut=std::strncmp(argv[7],"lut_",4)==0;
 std::vector<uint16_t> input(k*m);std::vector<float> table(129),scale(m,-1.f);
 std::vector<uint8_t> q(k*m,0xcd),elf(1<<20);read(argv[3],input);read(argv[4],table);
 Tensor inputs[]={tensor(DATA_BF16,k,m),tensor(DATA_F32,129)},outputs[]={tensor(DATA_F8_143,k,m),tensor(DATA_F32,1,m)};
 TensorAccessPattern in_ap[2]{},out_ap[2]{};AuxTensor auxiliary[16]{};
 HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=lut?2:1;in.outputTensorNr=2;
 in.inputTensors=inputs;in.outputTensors=outputs;std::string guid=std::string("fp8_fp_quant_")+(baseline?"amax16":argv[7]);std::strcpy(in.guid.name,guid.c_str());
 out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;out.auxiliaryTensors=auxiliary;
 out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=InstantiateTpcKernel(&in,&out);if(status!=GLUE_SUCCESS){std::cerr<<"glue error "<<status<<"\n";return 5;}
 if(baseline){std::ifstream f(argv[8],std::ios::binary|std::ios::ate);if(!f)return 6;out.kernel.elfSize=f.tellg();if(out.kernel.elfSize>elf.size())return 7;f.seekg(0);f.read(reinterpret_cast<char*>(elf.data()),out.kernel.elfSize);if(!f)return 8;}
 std::vector<tpc_tests::TensorDesc2> desc={descriptor(input.data(),k,1,m)};
 if(lut)desc.push_back(descriptor(table.data(),129,2));
 desc.push_back(descriptor(q.data(),k,0,m));desc.push_back(descriptor(scale.data(),1,2,m));
 VPEStats stats;std::cout<<"TPC_RUNNER=0; entering ISA simulation only"<<std::endl;
 unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);
 write(argv[5],q);write(argv[6],scale);
 std::cout<<"{\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"runtime_device_verified\":false}"<<std::endl;
 return 0;
}
