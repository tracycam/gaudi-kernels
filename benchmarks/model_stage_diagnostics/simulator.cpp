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
int main(int argc,char**argv){if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;std::string mode=argv[1],in=argv[2],out=argv[3];
 if(mode=="quant") {auto x=read<uint16_t>(in+"/input-bf16.bin",6144);std::vector<uint8_t>q(6144);std::vector<float>s(48);
 run("gk_block128_quant_v1",{{x.data(),DATA_BF16,1,{6144,1}}},{{q.data(),DATA_F8_143,0,{128,1,48}},{s.data(),DATA_F32,2,{1,1,48}}},{});
 write(out+"/isa-quant-fp8.bin",q);write(out+"/isa-quant-scales-f32.bin",s);std::cout<<"{\"mode\":\"quant\",\"outputs\":6144,\"device_accessed\":false}\n";
 }else {auto p=read<float>(in+"/partial-"+mode+".bin",48);auto as=read<float>(in+"/cpu-activation-scales-f32.bin",48),ws=read<float>(in+"/native-weight-scales-f32.bin",48);std::vector<float>part(48*128),bias(128);std::vector<uint16_t>y(128);for(int g=0;g<48;++g)for(int n=0;n<128;++n)part[g*128+n]=p[g];
 run("gk_block128_reduce_bias_v1",{{part.data(),DATA_F32,2,{128,1,48}},{as.data(),DATA_F32,2,{1,1,48}},{ws.data(),DATA_F32,2,{48,1}},{bias.data(),DATA_F32,2,{128}}},{{y.data(),DATA_BF16,1,{128,1}}},{});
 write(out+"/isa-reduce-"+mode+"-bf16.bin",y);for(auto v:y)if(v!=y[0])return 4;std::cout<<"{\"mode\":\""<<mode<<"\",\"replicated_bits\":"<<y[0]<<",\"device_accessed\":false}\n";}
 return 0;
}
