// Local instruction simulator only. No simulator implementation is vendored.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
static Tensor tensor(TensorDataType type,unsigned width,unsigned rows=1){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;for(unsigned d=0;d<MAX_TENSOR_DIM;++d)t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d==0?width:d==1?rows:1;t.quantizationParam.scale=1;return t;}
static tpc_tests::TensorDesc2 descriptor(void*data,unsigned width,unsigned log2,unsigned rows=1){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=reinterpret_cast<uint64_t>(data);d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}return d;}
static uint16_t bf(float x){uint32_t bits;std::memcpy(&bits,&x,4);return bits>>16;}
int main(int argc,char**argv){
 if(argc!=2)return 2;const char*runner=std::getenv("TPC_RUNNER");if(!runner||std::strcmp(runner,"0"))return 3;
 std::string mode=argv[1];bool history=mode=="history-safe"||mode=="history-unsafe",unsafe=mode=="history-unsafe";
 if(!history&&mode!="exact-subnormal"&&mode!="exact-significant"&&mode!="exact-cancellation"&&mode!="exact-negative-half")return 4;
 unsigned k=history?32:mode=="exact-cancellation"?3:1,n=history?512:1;uint32_t expected=history?(unsafe?0:0x42000000):(mode=="exact-subnormal"?0x00010000:mode=="exact-negative-half"?0x80000000:0x3b800000);
 std::vector<uint8_t>weights(history?8192:64,history?0x22:2),scales(history?512:64,mode=="exact-significant"?252:127),elf(1<<20);
 std::vector<uint16_t>activation(history?32:32,history?0x3f80:1),table(512),flag(32,unsafe);std::vector<int32_t>mapping(16,0);std::vector<uint32_t>output(n,0xdeadbeef);
 if(mode=="exact-cancellation"){weights[0]=0x27;weights[1]=0x0f;scales[0]=252;activation[0]=activation[2]=0x7f7f;}
 if(mode=="exact-negative-half"){weights[0]=1;scales[0]=111;activation[0]=0x8001;}
 const float q[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};for(int i=0;i<256;++i){table[2*i]=bf(q[i&15]);table[2*i+1]=bf(q[i>>4]);}
 std::vector<Tensor>inputs;std::vector<tpc_tests::TensorDesc2>desc;
 if(history){inputs={tensor(DATA_PACKED_MXFP4,256,32),tensor(DATA_U8,512),tensor(DATA_BF16,32),tensor(DATA_BF16,512),tensor(DATA_I32,1),tensor(DATA_U16,1)};
  desc={descriptor(weights.data(),256,0,32),descriptor(scales.data(),512,0),descriptor(activation.data(),32,1),descriptor(table.data(),512,1),descriptor(mapping.data(),1,2),descriptor(flag.data(),1,1)};
 }else{inputs={tensor(DATA_U8,(k+1)/2),tensor(DATA_U8,1),tensor(DATA_BF16,k)};desc={descriptor(weights.data(),(k+1)/2,0),descriptor(scales.data(),1,0),descriptor(activation.data(),k,1)};}
 Tensor outputs[]={tensor(DATA_F32,n)};desc.push_back(descriptor(output.data(),n,2));
 TensorAccessPattern in_ap[6]{},out_ap[1]{};AuxTensor auxiliary[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=outputs;
 std::strcpy(in.guid.name,history?"gk_mxfp4_dispatch_history":"gk_mxfp4_integer_exact_v3");int params[]={int(n),int(k)};if(!history){in.nodeParams.nodeParams=params;in.nodeParams.nodeParamsSize=8;}
 out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;out.auxiliaryTensors=auxiliary;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=InstantiateTpcKernel(&in,&out);if(status!=GLUE_SUCCESS){std::cerr<<"glue error "<<status<<"\n";return 5;}
 VPEStats stats;std::cout<<"TPC_RUNNER=0; instruction simulation only"<<std::endl;unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);
 unsigned bad=0;for(auto value:output)bad+=value!=expected;
 unsigned weight_loads=stats.scalarLoadPerTensor[0]+stats.vectorLoadPerTensor[0],scale_loads=stats.scalarLoadPerTensor[1]+stats.vectorLoadPerTensor[1];
 std::cout<<"{\"case\":\""<<mode<<"\",\"checked\":"<<n<<",\"bad\":"<<bad<<",\"first_bits\":"<<output[0]<<",\"expected_bits\":"<<expected<<",\"weight_loads\":"<<weight_loads<<",\"scale_loads\":"<<scale_loads<<",\"simulator_cycles\":"<<cycles<<",\"device_validated\":false}"<<std::endl;
 return bad||(unsafe&&(weight_loads||scale_loads))?6:0;
}
