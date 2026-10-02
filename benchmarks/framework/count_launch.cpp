// Read-only API counter. Counts outer calls only; does not intercept/replay work.
#include <synapse_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
static std::atomic<unsigned long long> launches{0};
static thread_local unsigned depth=0;
struct Guard{Guard(){if(depth++==0)++launches;}~Guard(){--depth;}};
template<class T>T real(const char* name){auto p=dlsym(RTLD_NEXT,name);if(!p){auto h=dlopen("/usr/lib/habanalabs/libSynapse.so",RTLD_NOW|RTLD_NOLOAD);if(h)p=dlsym(h,name);}if(!p){std::fprintf(stderr,"missing %s\n",name);std::abort();}return reinterpret_cast<T>(p);}
extern "C" unsigned long long gaudi_kernel_launch_count(){return launches.load();}
extern "C" synStatus synLaunch(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,uint32_t f){
 static auto fn=real<decltype(&synLaunch)>("synLaunch");Guard g;return fn(s,t,n,w,r,f);
}
extern "C" synStatus synLaunchWithExternalEvents(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*e,const uint32_t ne,uint32_t f){
 static auto fn=real<decltype(&synLaunchWithExternalEvents)>("synLaunchWithExternalEvents");Guard g;return fn(s,t,n,w,r,e,ne,f);
}
extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*e,const uint32_t ne,uint32_t f){
 static auto fn=real<decltype(&synLaunchWithExternalEventsExt)>("synLaunchWithExternalEventsExt");Guard g;return fn(s,t,n,w,r,e,ne,f);
}
