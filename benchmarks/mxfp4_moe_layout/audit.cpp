#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
struct View {void*data;TensorDataType type;unsigned log2;std::vector<unsigned>shape;};
static uint16_t bf(float x){uint32_t u;std::memcpy(&u,&x,4);return u>>16;}
static VPEStats run(const char*guid,const std::vector<View>&inputs,const std::vector<View>&outputs,std::vector<int>params){
 std::vector<Tensor>in,out;std::vector<tpc_tests::TensorDesc2>desc;
 auto add=[&](const View&v,std::vector<Tensor>&target){Tensor t{};t.geometry.dims=v.shape.size();t.geometry.dataType=v.type;t.quantizationParam.scale=1;tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=uint64_t(v.data);d.configuration=v.log2|(((1u<<v.shape.size())-1)<<8)|((v.shape.size()-1)<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i<v.shape.size()?v.shape[i]:1;t.geometry.maxSizes[i]=t.geometry.minSizes[i]=size;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}target.push_back(t);desc.push_back(d);};for(auto&v:inputs)add(v,in);for(auto&v:outputs)add(v,out);
 TensorAccessPattern ia[8]{},oa[4]{};AuxTensor aux[16]{};std::vector<uint8_t>elf(1<<20);HabanaKernelParams p{};HabanaKernelInstantiation q{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=in.size();p.outputTensorNr=out.size();p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,guid);p.nodeParams.nodeParams=params.data();p.nodeParams.nodeParamsSize=params.size()*4;q.inputTensorAccessPattern=ia;q.outputTensorAccessPattern=oa;q.auxiliaryTensors=aux;q.kernel.kernelElf=elf.data();q.kernel.elfSize=elf.size();if(InstantiateTpcKernel(&p,&q)!=GLUE_SUCCESS)throw 1;VPEStats stats;tpc_tests::RunSimulation(p,q,desc,stats);return stats;
}

#include <fstream>
#include <string>
static std::vector<uint8_t> read(const std::string&file){std::ifstream f(file,std::ios::binary|std::ios::ate);if(!f)throw 9;auto n=f.tellg();f.seekg(0);std::vector<uint8_t>v(n);f.read((char*)v.data(),n);return v;}
int main(int argc,char**argv){
 if(argc!=2||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::string dir=argv[1];unsigned N,K,E,splits;std::ifstream(dir+"/config.txt")>>N>>K>>E>>splits;
 auto old_w=read(dir+"/historical_weight.bin"),old_s=read(dir+"/historical_scale.bin"),new_w=read(dir+"/native_weight.bin"),new_s=read(dir+"/native_scale.bin"),expected=read(dir+"/expected_decoded.bf16"),x=read(dir+"/activation.bf16"),raw=read(dir+"/expected_partial.f32");
 std::vector<uint16_t>lut(512),decoded(N*K,0);float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};for(unsigned i=0;i<256;++i){lut[2*i]=bf(values[i&15]);lut[2*i+1]=bf(values[i>>4]);}
 int32_t selected=1;unsigned bad=0,checked=0,wrong_layout_bad=0;VPEStats old_stats;
 for(int layout=0;layout<3;++layout){bool native=layout==1,wrong=layout==2;auto&w=(native||wrong)?new_w:old_w;auto&s=(native||wrong)?new_s:old_s;
  auto stats=run(native?"layout_native":"layout_historical",{{w.data(),DATA_U8,0,native?std::vector<unsigned>{256,K,E*(N/512)}:std::vector<unsigned>{256,K*E*(N/512)}},{s.data(),DATA_U8,0,native?std::vector<unsigned>{512,K/32,E*(N/512)}:std::vector<unsigned>{512,(K/32)*E*(N/512)}},{lut.data(),DATA_BF16,1,{512,1}},{&selected,DATA_I32,2,{1,1}}},{{decoded.data(),DATA_BF16,1,{N,K,1}}},{int(N/512),0,0});
  if(!native&&!wrong)old_stats=stats;
  for(unsigned i=0;i<decoded.size();++i){bool mismatch=decoded[i]!=((uint16_t*)expected.data())[i];if(wrong)wrong_layout_bad+=mismatch;else{bad+=mismatch;++checked;}}
 }
 std::vector<float>partial(N*splits,-999);auto stats=run(splits==3?"layout_gp":"layout_down",{{old_w.data(),DATA_U8,0,{256,K*E*(N/512)}},{old_s.data(),DATA_U8,0,{512,(K/32)*E*(N/512)}},{x.data(),DATA_BF16,1,{K,1}},{lut.data(),DATA_BF16,1,{512,1}},{&selected,DATA_I32,2,{1,1}}},{{partial.data(),DATA_F32,2,{N*splits,1}}},{});
 for(unsigned part=0;part<splits;++part)for(unsigned n=0;n<N;++n){unsigned physical=(n/128)*128+(n%2)*64+(n%128)/2;float want=((float*)raw.data())[part*N+n];bad+=partial[part*N+physical]!=want;++checked;}
 if(!wrong_layout_bad)++bad;
 std::cout<<"{\"N\":"<<N<<",\"K\":"<<K<<",\"checked\":"<<checked<<",\"bad\":"<<bad<<",\"wrong_layout_decode_mismatches\":"<<wrong_layout_bad<<",\"legacy_decoder_packed_vector_loads\":"<<old_stats.vectorLoadPerTensor[0]<<",\"legacy_decoder_scale_vector_loads\":"<<old_stats.vectorLoadPerTensor[1]<<",\"old_ASM_weight_vector_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"old_ASM_lane_order\":\"even64 then odd64 in every logical128; GP preserves3K2048partials\",\"device_validated\":false}\n";
 return bad?3:0;
}
