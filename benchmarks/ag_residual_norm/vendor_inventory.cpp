// CPU-only public GUID inventory. No Synapse/device API, tensor, or acquisition.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
int main(int argc,char**argv){
 if(argc!=3)return 2;void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<'\n';return 3;}
 auto get=reinterpret_cast<pfnGetKernelGuids>(dlsym(lib,"GetKernelGuids"));if(!get)return 4;
 uint32_t count=0;auto status=get(DEVICE_ID_GAUDI2,&count,nullptr);if(status!=GLUE_SUCCESS||count>100000||count==0)return 5;
 std::vector<GuidInfo>guids(count);status=get(DEVICE_ID_GAUDI2,&count,guids.data());if(status!=GLUE_SUCCESS)return 6;
 std::ofstream f(argv[2]);unsigned relevant=0;for(auto&g:guids){f<<g.name<<'\n';if(std::strstr(g.name,"rms")||std::strstr(g.name,"fused_kernel_")){std::cout<<g.name<<'\n';++relevant;}}
 std::cout<<"{\"total_guids\":"<<count<<",\"rms_or_dynamic_fused_guids\":"<<relevant<<",\"device_acquired\":false}\n";return f?0:7;
}
