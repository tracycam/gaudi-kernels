// CPU-only public glue type gate. No Synapse/runtime/device entry points.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <iostream>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned width){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;for(int i=0;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i?1:width;return t;}
int main(int argc,char**argv){
 if(argc!=2)return 2;void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h){std::cerr<<dlerror();return 3;}
 using Fn=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
 auto instantiate=reinterpret_cast<Fn>(dlsym(h,"InstantiateTpcKernel"));if(!instantiate)return 4;
 Tensor inputs[]={tensor(DATA_F32,384),tensor(DATA_I32,8)},outputs[]={tensor(DATA_F32,8),tensor(DATA_I32,8),tensor(DATA_F32,1)};
 TensorAccessPattern in_ap[2]{},out_ap[3]{};struct Params{float factor;int renormalize;int apply_scale;}params{1.f,1,0};
 HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=2;in.outputTensorNr=3;in.inputTensors=inputs;in.outputTensors=outputs;in.nodeParams={&params,sizeof(params)};out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;
 for(const char*name:{"gk_router_post8_scalar_experiment","gk_router_post8_vector_experiment"}){
  std::strcpy(in.guid.name,name);inputs[1].geometry.dataType=DATA_I32;out.kernel.elfSize=0;
  auto valid=instantiate(&in,&out);if(valid!=GLUE_INSUFFICIENT_ELF_BUFFER)return 5;
  inputs[1].geometry.dataType=DATA_I64;auto rejected=instantiate(&in,&out);if(rejected!=GLUE_INCOMPATIBLE_DATA_TYPE)return 6;
  std::cout<<"{\"guid\":\""<<name<<"\",\"physical_I32_accepted\":true,\"physical_I64_rejected\":true,\"device_used\":false}\n";
 }
 dlclose(h);
}
