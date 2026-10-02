// Read-only CPU inspection of a sealed TPC library. Does not load Synapse or acquire a device.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,std::initializer_list<unsigned> dims) {
 Tensor t{}; t.geometry.dataType=type; t.geometry.dims=dims.size();
 unsigned d=0;for(auto x:dims)t.geometry.maxSizes[d]=t.geometry.minSizes[d]=x,++d;
 return t;
}
static void dump(const TensorAccessPattern& a,unsigned dims) {
 std::cout<<"{\"flags\":"<<a.Value<<",\"mapping\":[";
 for(unsigned d=0;d<dims;++d){auto&m=a.mapping[d];if(d)std::cout<<',';
  std::cout<<"{\"index\":"<<m.indexSpaceDim<<",\"a\":"<<m.a<<",\"start\":"<<m.start_b<<",\"end\":"<<m.end_b<<",\"allRequired\":"<<m.allRequired<<'}';}
 std::cout<<"]}";
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h){std::cerr<<dlerror();return 3;}
 auto fn=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(h,"InstantiateTpcKernel"));if(!fn)return 4;
 Tensor in[]={tensor(DATA_F32,{3392,1,48}),tensor(DATA_F32,{1,1,48}),tensor(DATA_F32,{48,27}),tensor(DATA_F32,{3392})};
 Tensor out[]={tensor(DATA_BF16,{3392,1})};TensorAccessPattern ia[4]{},oa[1]{};AuxTensor aux[16]{};
 std::vector<unsigned char>elf(1<<20);HabanaKernelParams p{};HabanaKernelInstantiation q{};
 p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=4;p.outputTensorNr=1;p.inputTensors=in;p.outputTensors=out;
 std::strncpy(p.guid.name,argv[2],sizeof(p.guid.name)-1);q.inputTensorAccessPattern=ia;q.outputTensorAccessPattern=oa;q.auxiliaryTensors=aux;q.kernel.kernelElf=elf.data();q.kernel.elfSize=elf.size();
 auto status=fn(&p,&q);if(status!=GLUE_SUCCESS){std::cerr<<"instantiate="<<status;return 5;}
 std::ofstream f(argv[3],std::ios::binary);f.write(reinterpret_cast<char*>(elf.data()),q.kernel.elfSize);if(!f)return 6;
 std::cout<<"{\"device_accessed\":false,\"guid\":\""<<argv[2]<<"\",\"elf_bytes\":"<<q.kernel.elfSize<<",\"index_geometry\":[";
 for(unsigned d=0;d<q.indexSpaceRank;++d){if(d)std::cout<<',';std::cout<<q.indexSpaceGeometry[d];}
 std::cout<<"],\"inputs\":[";for(unsigned i=0;i<4;++i){if(i)std::cout<<',';dump(ia[i],in[i].geometry.dims);}
 std::cout<<"],\"outputs\":[";dump(oa[0],2);std::cout<<"]}\n";
}
