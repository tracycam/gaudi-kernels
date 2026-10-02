// Offline TPC ISA only, using public tensor descriptors and the actual glue.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tpc_lib_api;
using I=std::vector<int>;
Tensor tensor(unsigned n,unsigned m=1,bool matrix=false){Tensor t{};t.geometry.dims=matrix?2:1;t.geometry.dataType=DATA_I32;for(int i=0;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?n:i==1?m:1;t.quantizationParam.scale=1;return t;}
tpc_tests::TensorDesc2 desc(I&v,unsigned n,unsigned m=1){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)v.data();d.configuration=2u|(3u<<8)|(1u<<16);unsigned stride=1;for(int i=0;i<5;++i){auto size=i==0?n:i==1?m:1;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}return d;}
void save(const std::string&path,const I&v){std::ofstream f(path,std::ios::binary);f.write((char*)v.data(),v.size()*4);if(!f)throw std::runtime_error("write failed");}
using Instantiate=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
void run(Instantiate fn,const char*name,std::vector<Tensor>in,std::vector<Tensor>out,std::vector<int>params,std::vector<tpc_tests::TensorDesc2>d){
 HabanaKernelParams p{};HabanaKernelInstantiation k{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=in.size();p.outputTensorNr=out.size();p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,name);p.nodeParams.nodeParams=params.data();p.nodeParams.nodeParamsSize=params.size()*4;
 std::vector<TensorAccessPattern>ia(in.size()),oa(out.size());k.inputTensorAccessPattern=ia.data();k.outputTensorAccessPattern=oa.data();std::vector<unsigned char>elf(4<<20);k.kernel.kernelElf=elf.data();k.kernel.elfSize=elf.size();auto status=fn(&p,&k);if(status!=GLUE_SUCCESS)throw std::runtime_error("glue status "+std::to_string(status));
 VPEStats stats;unsigned cycles=tpc_tests::RunSimulation(p,k,d,stats);
 std::cout<<"{\"guid\":\""<<name<<"\",\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"TPC_RUNNER\":0}"<<std::endl;
}
int main(int argc,char**argv){
 if(argc!=9||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 int t=std::stoi(argv[4]),r=std::stoi(argv[5]),e=std::stoi(argv[6]),c=std::stoi(argv[7]),b=std::stoi(argv[8]);if(t<1||t>513||r<1||r>8||e<r||e>384||c<1||c>32||c>t||b<1||b>t*r)return 3;
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());auto fn=(Instantiate)dlsym(h,"InstantiateTpcKernel");if(!fn)return 4;
 I ids(t*r),counts(e,0x5a5a5a5a),flags(t,0x5a5a5a5a),prefix(e+1,0x5a5a5a5a),expert(b,0x5a5a5a5a),base(b,0x5a5a5a5a),valid(b,0x5a5a5a5a),status(1,0x5a5a5a5a),map(b*c,0x5a5a5a5a),inverse(t*r,0x5a5a5a5a);
 std::ifstream input(argv[2],std::ios::binary);input.read((char*)ids.data(),ids.size()*4);if(!input)return 5;
 run(fn,"gk_route_count_i32_v1",{tensor(r,t,true)},{tensor(e),tensor(t)},{e},{desc(ids,r,t),desc(counts,e),desc(flags,t)});
 run(fn,"gk_route_prefix_i32_v1",{tensor(e),tensor(t)},{tensor(e+1),tensor(b),tensor(b),tensor(b),tensor(1)},{r,c,b},{desc(counts,e),desc(flags,t),desc(prefix,e+1),desc(expert,b),desc(base,b),desc(valid,b),desc(status,1)});
 run(fn,"gk_route_scatter_i32_v1",{tensor(r,t,true),tensor(e+1),tensor(1)},{tensor(b*c),tensor(t*r)},{c,b},{desc(ids,r,t),desc(prefix,e+1),desc(status,1),desc(map,b*c),desc(inverse,t*r)});
 std::string out=argv[3];std::vector<std::pair<std::string,I*>>values={{"counts",&counts},{"row_status",&flags},{"prefix",&prefix},{"tile_expert",&expert},{"tile_base",&base},{"valid_rows",&valid},{"status",&status},{"row_map",&map},{"inverse",&inverse}};for(auto&p:values)save(out+"/"+p.first+".bin",*p.second);dlclose(h);
}
