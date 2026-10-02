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
 d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;
 for(unsigned dim=0;dim<5;++dim){unsigned size=dim<sizes.size()?sizes[dim]:1;d.dimDescriptors[dim].size=size;d.dimDescriptors[dim].stride=stride;stride*=size;}
 return d;
}
int main(int argc,char**argv){
 if(argc!=8){std::cerr<<"usage: simulator kind N K packed.bin scales.bin table.bin output.bin\n";return 2;}
 const char*mode=std::getenv("TPC_RUNNER");if(!mode||std::strcmp(mode,"0")){std::cerr<<"TPC_RUNNER must explicitly be0\n";return 3;}
 bool native=std::strcmp(argv[1],"native")==0;unsigned n=std::stoul(argv[2]),k=std::stoul(argv[3]);
 if((native&&(n!=512||k!=32))||(!native&&(n!=3||(k!=31&&k!=32))))return 4;
 std::vector<uint8_t>packed(n*((k+1)/2)),scales(n*((k+31)/32)),elf(1<<20);
 std::vector<uint16_t>table(512),output(n*k,0xcdcd);read(argv[4],packed);read(argv[5],scales);read(argv[6],table);
 std::vector<unsigned>wshape=native?std::vector<unsigned>{256,k,1}:std::vector<unsigned>{(k+1)/2,n};
 std::vector<unsigned>sshape=native?std::vector<unsigned>{512,k/32,1}:std::vector<unsigned>{(k+31)/32,n};
 std::vector<unsigned>yshape=native?std::vector<unsigned>{n,k}:std::vector<unsigned>{k,n};
 Tensor inputs[]={tensor(DATA_U8,wshape),tensor(DATA_U8,sshape),tensor(DATA_BF16,{512,1})},outputs[]={tensor(DATA_BF16,yshape)};
 TensorAccessPattern input_ap[3]{},output_ap[1]{};AuxTensor aux[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=3;in.outputTensorNr=1;in.inputTensors=inputs;in.outputTensors=outputs;
 std::strcpy(in.guid.name,native?"gk_mx4c3_native_s2_252_bf16":"gk_mx4c3_rows_s2_252_bf16");
 int params[2]={0,0};if(native){in.nodeParams.nodeParams=params;in.nodeParams.nodeParamsSize=8;}
 out.inputTensorAccessPattern=input_ap;out.outputTensorAccessPattern=output_ap;out.auxiliaryTensors=aux;
 out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=InstantiateTpcKernel(&in,&out);if(status!=GLUE_SUCCESS){std::cerr<<"glue error "<<status<<'\n';return 5;}
 std::vector<tpc_tests::TensorDesc2>desc={descriptor(packed.data(),wshape,0),descriptor(scales.data(),sshape,0),descriptor(table.data(),{512,1},1),descriptor(output.data(),yshape,1)};
 VPEStats stats;std::cout<<"TPC_RUNNER=0; entering ISA simulation only"<<std::endl;
 unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);write(argv[7],output);
 std::cout<<"{\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"runtime_device_verified\":false}"<<std::endl;
}
