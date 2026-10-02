// CPU descriptor/ELF differential only; no simulation or device API.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tpc_lib_api;
using Fn=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
Tensor tensor(TensorDataType type,std::initializer_list<unsigned>dims){Tensor t{};t.geometry.dataType=type;t.geometry.dims=dims.size();unsigned i=0;for(unsigned n:dims)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=n,++i;for(;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=1;return t;}
struct Case{const char*name;std::vector<Tensor>in,out;std::vector<int>params;};
struct Result{GlueCodeReturn status;HabanaKernelInstantiation k{};std::vector<TensorAccessPattern>ia,oa;std::vector<unsigned char>elf;};
Result run(Fn fn,Case c,bool wide){Result r;r.ia.resize(c.in.size());r.oa.resize(c.out.size());r.elf.resize(1<<20);r.k.inputTensorAccessPattern=r.ia.data();r.k.outputTensorAccessPattern=r.oa.data();r.k.kernel.kernelElf=r.elf.data();r.k.kernel.elfSize=r.elf.size();HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=c.in.size();p.outputTensorNr=c.out.size();p.inputTensors=c.in.data();p.outputTensors=c.out.data();p.nodeParams.nodeParams=c.params.data();p.nodeParams.nodeParamsSize=c.params.size()*4;std::string name=c.name;if(wide)name.replace(0,9,"gk_route_wide_");std::strcpy(p.guid.name,name.c_str());r.status=fn(&p,&r.k);r.elf.resize(r.k.kernel.elfSize);return r;}
void need(bool b){if(!b)throw std::runtime_error("descriptor gate failed");}
int main(int argc,char**argv){if(argc!=4)return 2;auto load=[](const char*p){auto h=dlopen(p,RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());return reinterpret_cast<Fn>(dlsym(h,"InstantiateTpcKernel"));};Fn wide=load(argv[1]),meta=load(argv[2]),tiles=load(argv[3]);unsigned checks=0;
for(unsigned c:{32u,64u,128u,129u}){unsigned t=513,r=8,e=384,b=500,q=(t*r+63)/64+1;std::vector<Case>cases={
 {"gk_route_count_i32_v3",{tensor(DATA_I32,{r,t}),tensor(DATA_I32,{t*r})},{tensor(DATA_I32,{e}),tensor(DATA_I32,{t}),tensor(DATA_I32,{q,e})},{int(e)}},
 {"gk_route_prefix_i32_v3",{tensor(DATA_I32,{e}),tensor(DATA_I32,{t})},{tensor(DATA_I32,{e+1}),tensor(DATA_I32,{b}),tensor(DATA_I32,{b}),tensor(DATA_I32,{b}),tensor(DATA_I32,{1})},{int(r),int(c),int(b)}},
 {"gk_route_inverse_i32_v3",{tensor(DATA_I32,{r,t}),tensor(DATA_I32,{t*r}),tensor(DATA_I32,{e+1}),tensor(DATA_I32,{1}),tensor(DATA_I32,{q,e})},{tensor(DATA_I32,{t*r})},{int(c),int(b)}},
 {"gk_route_row_map_i32_v3",{tensor(DATA_I32,{t*r}),tensor(DATA_I32,{b}),tensor(DATA_I32,{1})},{tensor(DATA_I32,{b*c})},{int(c)}},
 {"gk_route_tile_gather_bf16_v1",{tensor(DATA_BF16,{6144,t}),tensor(DATA_I32,{b*c}),tensor(DATA_I32,{1})},{tensor(DATA_BF16,{6144,c,2})},{int(r),int(c),1}},
 {"gk_route_tile_gate_bf16_v1",{tensor(DATA_F32,{512,c,2}),tensor(DATA_I32,{b}),tensor(DATA_I32,{1})},{tensor(DATA_BF16,{256,c,2})},{1}},
 {"gk_route_tile_combine_f32_v1",{tensor(DATA_F32,{512,b*c}),tensor(DATA_F32,{r,t}),tensor(DATA_I32,{t*r}),tensor(DATA_I32,{1})},{tensor(DATA_F32,{512,t})},{}}
 };for(unsigned i=0;i<cases.size();++i){auto a=run(wide,cases[i],true);bool valid=c<=128||i==0||i==6;need((a.status==GLUE_SUCCESS)==valid);++checks;if(!valid)continue;
 if(c==32){auto old=run(i<4?meta:tiles,cases[i],false);need(old.status==a.status&&old.elf==a.elf);need(old.k.indexSpaceRank==a.k.indexSpaceRank&&!std::memcmp(old.k.indexSpaceGeometry,a.k.indexSpaceGeometry,sizeof(a.k.indexSpaceGeometry)));need(old.k.kernel.paramsNr==a.k.kernel.paramsNr&&!std::memcmp(old.k.kernel.scalarParams,a.k.kernel.scalarParams,sizeof(a.k.kernel.scalarParams)));need(!std::memcmp(old.ia.data(),a.ia.data(),a.ia.size()*sizeof(TensorAccessPattern))&&!std::memcmp(old.oa.data(),a.oa.data(),a.oa.size()*sizeof(TensorAccessPattern)));++checks;}
 if(i==4)need(a.k.indexSpaceGeometry[0]==c&&a.k.indexSpaceGeometry[1]==48&&a.k.indexSpaceGeometry[2]==2);
 if(i==5)need(a.k.indexSpaceGeometry[0]==2&&a.k.indexSpaceGeometry[1]==c&&a.k.indexSpaceGeometry[2]==2);
 }}std::cout<<"{\"status\":\"PASS_CPU_GLUE_DIFFERENTIAL\",\"checks\":"<<checks<<",\"device_acquired\":false}"<<std::endl;}
