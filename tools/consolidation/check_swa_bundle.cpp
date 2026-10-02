// Compare original and bundled glue output and returned TPC ELF bytes; no HPU.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cassert>
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
using namespace tpc_lib_api;
using Guids=GlueCodeReturn(*)(DeviceId,uint32_t*,GuidInfo*);
using Instantiate=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
static void shape(Tensor& t,TensorDataType dtype,std::initializer_list<unsigned long> dims){
  t.geometry.dataType=dtype;t.geometry.dims=dims.size();unsigned i=0;
  for(auto n:dims){t.geometry.minSizes[i]=t.geometry.maxSizes[i]=n;++i;}
}
int main(int argc,char**argv){
  assert(argc==4);
  void* av=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  void* batch=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
  void* bundle=dlopen(argv[3],RTLD_NOW|RTLD_LOCAL);assert(av&&batch&&bundle);
  auto merged=(Instantiate)dlsym(bundle,"InstantiateTpcKernel");assert(merged);
  auto guids=(Guids)dlsym(bundle,"GetKernelGuids");assert(guids);
  uint32_t n=0;assert(guids(DEVICE_ID_GAUDI2,&n,nullptr)==GLUE_SUCCESS&&n==6);
  GuidInfo names[6]{};n=6;assert(guids(DEVICE_ID_GAUDI2,&n,names)==GLUE_SUCCESS&&n==6);
  unsigned count=0;
  for(auto& guid:names){
    bool is_batch=std::strstr(guid.name,"batch")!=nullptr;
    bool debug=std::strstr(guid.name,"debug")!=nullptr;
    auto original=(Instantiate)dlsym(is_batch?batch:av,"InstantiateTpcKernel");assert(original);
    auto params=std::make_unique<HabanaKernelParams>();
    auto& in=*params;std::memset(&in,0,sizeof(in));
    std::array<Tensor,7> inputs{};std::array<Tensor,4> outputs{};
    in.inputTensors=inputs.data();in.outputTensors=outputs.data();
    in.deviceId=DEVICE_ID_GAUDI2;in.guid=guid;in.inputTensorNr=7;in.outputTensorNr=debug?4:1;
    float scale=.07216878f;in.nodeParams.nodeParams=&scale;in.nodeParams.nodeParamsSize=4;
    unsigned rows=is_batch?3:1;
    shape(in.inputTensors[0],DATA_BF16,{3072,1,rows});
    shape(in.inputTensors[1],DATA_BF16,{192,1,256});
    shape(in.inputTensors[2],DATA_BF16,{128,1,256});
    shape(in.inputTensors[3],DATA_I32,{2});shape(in.inputTensors[4],DATA_I32,{2});
    shape(in.inputTensors[5],DATA_I32,{rows});shape(in.inputTensors[6],DATA_BF16,{16});
    shape(in.outputTensors[0],DATA_BF16,{128,16,rows});
    for(unsigned i=1;i<in.outputTensorNr;++i)shape(in.outputTensors[i],DATA_F32,{128,16,rows});
    auto first=std::make_unique<HabanaKernelInstantiation>();
    auto second=std::make_unique<HabanaKernelInstantiation>();
    auto& a=*first;auto& b=*second;std::memset(&a,0,sizeof(a));std::memset(&b,0,sizeof(b));
    std::array<TensorAccessPattern,7> ai{},bi{};
    std::array<TensorAccessPattern,4> ao{},bo{};
    a.inputTensorAccessPattern=ai.data();a.outputTensorAccessPattern=ao.data();
    b.inputTensorAccessPattern=bi.data();b.outputTensorAccessPattern=bo.data();
    assert(original(&in,&a)==GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(merged(&in,&b)==GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(a.kernel.elfSize==b.kernel.elfSize&&a.kernel.elfSize>0);
    std::vector<unsigned char> ea(a.kernel.elfSize),eb(b.kernel.elfSize);
    a.kernel.kernelElf=ea.data();b.kernel.kernelElf=eb.data();
    assert(original(&in,&a)==GLUE_SUCCESS);assert(merged(&in,&b)==GLUE_SUCCESS);
    assert(ea==eb);a.kernel.kernelElf=b.kernel.kernelElf=nullptr;
    assert(!std::memcmp(ai.data(),bi.data(),sizeof(ai))&&!std::memcmp(ao.data(),bo.data(),sizeof(ao)));
    a.inputTensorAccessPattern=b.inputTensorAccessPattern=nullptr;
    a.outputTensorAccessPattern=b.outputTensorAccessPattern=nullptr;
    assert(!std::memcmp(&a,&b,sizeof(a)));
    std::printf("%s ELF and access pattern equal (%u bytes)\n",guid.name,a.kernel.elfSize);++count;
  }
  assert(count==6);
}
