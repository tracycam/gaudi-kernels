#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
struct View {void*data;TensorDataType type;unsigned log2;std::vector<unsigned>shape;};
static uint16_t bf(float x){uint32_t u;std::memcpy(&u,&x,4);return u>>16;}
static VPEStats run(const char*guid,const std::vector<View>&inputs,const std::vector<View>&outputs,std::vector<int>params){
 std::vector<Tensor>in,out;std::vector<tpc_tests::TensorDesc2>desc;
 auto add=[&](const View&v,std::vector<Tensor>&target){Tensor t{};t.geometry.dims=v.shape.size();t.geometry.dataType=v.type;t.quantizationParam.scale=1;tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=uint64_t(v.data);d.configuration=v.log2|(((1u<<v.shape.size())-1)<<8)|((v.shape.size()-1)<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i<v.shape.size()?v.shape[i]:1;t.geometry.maxSizes[i]=t.geometry.minSizes[i]=size;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}target.push_back(t);desc.push_back(d);};for(auto&v:inputs)add(v,in);for(auto&v:outputs)add(v,out);
 TensorAccessPattern ia[8]{},oa[4]{};AuxTensor aux[16]{};std::vector<uint8_t>elf(1<<20);HabanaKernelParams p{};HabanaKernelInstantiation q{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=in.size();p.outputTensorNr=out.size();p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,guid);p.nodeParams.nodeParams=params.data();p.nodeParams.nodeParamsSize=params.size()*4;q.inputTensorAccessPattern=ia;q.outputTensorAccessPattern=oa;q.auxiliaryTensors=aux;q.kernel.kernelElf=elf.data();q.kernel.elfSize=elf.size();if(InstantiateTpcKernel(&p,&q)!=GLUE_SUCCESS)throw 1;VPEStats stats;tpc_tests::RunSimulation(p,q,desc,stats);return stats;
}

static float f32(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}



template<class T>static std::vector<T> read(const std::string&p,size_t n){std::vector<T>v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*sizeof(T));if(!f)throw 2;return v;}
template<class T>static void write(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));if(!f)throw 3;}

template<class T>static int check(const char*guid,const std::string&in,const std::string&out,int m,int n,int g){
 auto part=read<float>(in+"/partial.bin",g*m*n),as=read<float>(in+"/activation-scales.bin",g*m),ws=read<float>(in+"/weight-scales.bin",((n+127)/128)*g),bias=read<float>(in+"/bias.bin",n);
 bool fp32=sizeof(T)==4;auto expected=read<T>(in+(fp32?"/cpu-chain0-f32.bin":"/cpu-chain0-bf16.bin"),m*n);std::vector<T>y(m*n);
 auto stats=run(guid,{{part.data(),DATA_F32,2,{(unsigned)n,(unsigned)m,(unsigned)g}},{as.data(),DATA_F32,2,{1,(unsigned)m,(unsigned)g}},{ws.data(),DATA_F32,2,{(unsigned)g,(unsigned)((n+127)/128)}},{bias.data(),DATA_F32,2,{(unsigned)n}}},{{y.data(),fp32?DATA_F32:DATA_BF16,fp32?2u:1u,{(unsigned)n,(unsigned)m}}},{});
 write(out,y);unsigned bad=0;for(unsigned i=0;i<y.size();++i)bad+=y[i]!=expected[i];std::cout<<"{\"checked\":"<<y.size()<<",\"bit_mismatches_vs_declared_cpu_order\":"<<bad<<",\"instructions_executed\":"<<stats.instructionsExecuted<<",\"full_nops\":"<<stats.numFullNops<<",\"vpu_nops\":"<<stats.numVPUNops<<",\"load_transactions\":"<<stats.numLoadTransactions<<",\"store_transactions\":"<<stats.numStoreTransactions<<",\"output_bits\":"<<(fp32?32:16)<<",\"device_accessed\":false}\n";return bad?4:0;
}
int main(int argc,char**argv){
 if(argc!=7||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 int c=std::atoi(argv[1]),m=std::atoi(argv[3]),n=std::atoi(argv[4]),g=std::atoi(argv[5]);if(c<0||c>5||m<1||n<1||g<1)return 2;
 const char*guid[]={"gk_neumaier_isa_baseline_v1","gk_neumaier_isa_lookahead_v1","gk_neumaier_isa_handschedule_v1","gk_neumaier_isa_baseline_debug_v1","gk_neumaier_isa_lookahead_debug_v1","gk_neumaier_isa_handschedule_debug_v1"};
 return c>=3?check<uint32_t>(guid[c],argv[2],argv[6],m,n,g):check<uint16_t>(guid[c],argv[2],argv[6],m,n,g);
}
