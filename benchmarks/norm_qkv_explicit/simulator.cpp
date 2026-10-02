// Actual archived ELF execution; TPC_RUNNER=0 is mandatory. No HPU runtime.
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
template<class T> void read(const std::string& path,std::vector<T>& v) {
 std::ifstream f(path,std::ios::binary); f.read((char*)v.data(),v.size()*sizeof(T));
 if(!f || f.peek()!=EOF)throw std::runtime_error("wrong input size: "+path);
}
template<class T> void write(const std::string& path,const std::vector<T>& v) {
 std::ofstream f(path,std::ios::binary); f.write((char*)v.data(),v.size()*sizeof(T));
 if(!f)throw std::runtime_error("output failure: "+path);
}
static Tensor tensor(TensorDataType type,std::initializer_list<unsigned> sizes) {
 Tensor t{};t.geometry.dims=sizes.size();t.geometry.dataType=type;
 unsigned j=0;for(unsigned size:sizes)t.geometry.minSizes[j]=t.geometry.maxSizes[j]=size,++j;
 for(;j<MAX_TENSOR_DIM;++j)t.geometry.minSizes[j]=t.geometry.maxSizes[j]=1;
 t.quantizationParam.scale=1;t.quantizationParam.fp8bias=7;return t;
}
static tpc_tests::TensorDesc2 desc(void* ptr,unsigned log2,std::initializer_list<unsigned> sizes) {
 tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)ptr;
 // bits 16..18 identify the last active dimension (zero based). Hardcoding
 // rank two makes every [G,1,FCD] store alias group zero in the simulator.
 d.configuration=log2|(((1u<<sizes.size())-1u)<<8)|((sizes.size()-1u)<<16);unsigned j=0,stride=1;
 for(unsigned size:sizes){d.dimDescriptors[j].size=size;d.dimDescriptors[j].stride=stride;stride*=size;++j;}
 for(;j<5;++j){d.dimDescriptors[j].size=1;d.dimDescriptors[j].stride=stride;}return d;
}
int main(int argc,char** argv) {
 if(argc!=6)throw std::runtime_error("simulator LIB norm|block|fused WIDTH INPUT_DIR OUTPUT_DIR");
 if(!getenv("TPC_RUNNER")||std::strcmp(getenv("TPC_RUNNER"),"0"))return 3;
 std::string mode=argv[2],input=argv[4],output=argv[5];unsigned h=std::stoul(argv[3]),g=(h+127)/128;
 if(h==0||h>8192||(mode!="norm"&&mode!="block"&&mode!="fused"&&mode!="grid24"))return 4;
 void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
 auto instantiate=(GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*))dlsym(lib,"InstantiateTpcKernel");
 if(!instantiate)throw std::runtime_error("no InstantiateTpcKernel");
 std::vector<uint16_t> x(h),r(h),w(h),rr(h,0xdead),norm(h,0xdead);
 std::vector<uint8_t> q(g*128,0xcd),elf(1<<20);std::vector<float> scale(g,-1);
 std::vector<Tensor> inputs,outputs;std::vector<tpc_tests::TensorDesc2> descriptors;
 if(mode=="block") {
  read(input+"/norm.bin",x);inputs={tensor(DATA_BF16,{h,1})};
  outputs={tensor(DATA_F8_143,{128,1,g}),tensor(DATA_F32,{1,1,g})};
  descriptors={desc(x.data(),1,{h,1}),desc(q.data(),0,{128,1,g}),desc(scale.data(),2,{1,1,g})};
 } else {
  read(input+"/x.bin",x);read(input+"/residual.bin",r);read(input+"/gamma.bin",w);
  inputs={tensor(DATA_BF16,{h,1}),tensor(DATA_BF16,{h,1}),tensor(DATA_BF16,{h})};
  outputs={tensor(DATA_BF16,{h,1})};
  descriptors={desc(x.data(),1,{h,1}),desc(r.data(),1,{h,1}),desc(w.data(),1,{h}),desc(rr.data(),1,{h,1})};
  if(mode=="norm"){outputs.push_back(tensor(DATA_BF16,{h,1}));descriptors.push_back(desc(norm.data(),1,{h,1}));}
  else{outputs.push_back(tensor(DATA_F8_143,{128,1,g}));outputs.push_back(tensor(DATA_F32,{1,1,g}));descriptors.push_back(desc(q.data(),0,{128,1,g}));descriptors.push_back(desc(scale.data(),2,{1,1,g}));}
 }
 TensorAccessPattern ia[3]{},oa[3]{};AuxTensor aux[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=outputs.size();in.inputTensors=inputs.data();in.outputTensors=outputs.data();
 const char* guid=mode=="norm"?"gk_residual_rmsnorm_bf16_v1":mode=="block"?"gk_block128_quant_v1":mode=="grid24"?"gk_norm_block128_grid24_v1":"gk_residual_rmsnorm_block128_a8_v1";
 std::strcpy(in.guid.name,guid);float epsilon=1e-6f;
 in.nodeParams.nodeParams=mode=="block"?nullptr:&epsilon;in.nodeParams.nodeParamsSize=mode=="block"?0:4;
 out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;out.auxiliaryTensors=aux;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
 auto status=instantiate(&in,&out);if(status!=GLUE_SUCCESS)throw std::runtime_error("glue status "+std::to_string(status));
 std::vector<uint8_t> actual_elf(elf.begin(),elf.begin()+out.kernel.elfSize);write(output+"/kernel.o",actual_elf);
 VPEStats stats;std::cout<<"TPC_RUNNER=0; ISA simulation only"<<std::endl;
 auto cycles=tpc_tests::RunSimulation(in,out,descriptors,stats);
 if(mode!="block")write(output+"/residual.bin",rr);
 if(mode=="norm")write(output+"/norm.bin",norm);else{write(output+"/q.bin",q);write(output+"/scale.bin",scale);}
 std::cout<<"{\"simulator_cycles\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"index_space\":["<<out.indexSpaceGeometry[0]<<","<<out.indexSpaceGeometry[1]<<"],\"guid\":\""<<guid<<"\",\"device_verified\":false}"<<std::endl;
 dlclose(lib);
}
