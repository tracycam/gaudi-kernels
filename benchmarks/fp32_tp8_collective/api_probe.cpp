// Parameter/call audit only; never replays borrowed recipes or changes pointers.
#include <synapse_api.h>
#include <hccl.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
static std::mutex mu;static FILE*out=nullptr;static std::atomic<bool>active{false};static thread_local int depth=0;
struct Scope{bool outer;Scope():outer(depth++==0){}~Scope(){--depth;}};
template<class T>T real(const char*name){void*p=dlsym(RTLD_NEXT,name);if(!p){void*h=dlopen("/usr/lib/habanalabs/libSynapse.so",RTLD_NOW|RTLD_NOLOAD);if(h)p=dlsym(h,name);}if(!p){std::fprintf(stderr,"missing audit symbol %s\n",name);std::abort();}return reinterpret_cast<T>(p);}
static void row(const char*api,size_t count,int dtype,int op,int status,bool outer,const void*src=nullptr,const void*dst=nullptr,const void*stream=nullptr,const void*event=nullptr){if(!outer||!active.load())return;std::lock_guard<std::mutex>g(mu);if(out&&active.load())std::fprintf(out,"{\"api\":\"%s\",\"count\":%zu,\"dtype\":%d,\"reduce_op\":%d,\"status\":%d,\"send_address\":%llu,\"recv_address\":%llu,\"stream\":%llu,\"event\":%llu}\n",api,count,dtype,op,status,(unsigned long long)src,(unsigned long long)dst,(unsigned long long)stream,(unsigned long long)event);}
extern "C" int tp8_audit_begin(const char*path){std::lock_guard<std::mutex>g(mu);if(out)return 1;out=std::fopen(path,"w");if(!out)return 2;active=true;return 0;}
extern "C" int tp8_audit_end(){std::lock_guard<std::mutex>g(mu);active=false;if(!out)return 1;int rc=std::fclose(out);out=nullptr;return rc;}
extern "C" hcclResult_t hcclAllReduce(const void*s,void*d,size_t n,hcclDataType_t type,hcclRedOp_t op,hcclComm_t c,void*stream){static auto f=real<decltype(&hcclAllReduce)>("hcclAllReduce");Scope scope;auto rc=f(s,d,n,type,op,c,stream);row("hcclAllReduce",n,type,op,rc,scope.outer,s,d,stream);return rc;}
extern "C" hcclResult_t hcclAllGather(const void*s,void*d,size_t n,hcclDataType_t type,hcclComm_t c,void*stream){static auto f=real<decltype(&hcclAllGather)>("hcclAllGather");Scope scope;auto rc=f(s,d,n,type,c,stream);row("hcclAllGather",n,type,-1,rc,scope.outer,s,d,stream);return rc;}
extern "C" synStatus synLaunch(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,uint32_t flags){static auto f=real<decltype(&synLaunch)>("synLaunch");Scope scope;auto rc=f(s,t,n,w,r,flags);row("synLaunch",n,-1,-1,rc,scope.outer,nullptr,nullptr,s);return rc;}
extern "C" synStatus synLaunchWithExternalEvents(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*e,const uint32_t ne,uint32_t flags){static auto f=real<decltype(&synLaunchWithExternalEvents)>("synLaunchWithExternalEvents");Scope scope;auto rc=f(s,t,n,w,r,e,ne,flags);row("synLaunchWithExternalEvents",n,-1,-1,rc,scope.outer,nullptr,nullptr,s);return rc;}
extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*e,const uint32_t ne,uint32_t flags){static auto f=real<decltype(&synLaunchWithExternalEventsExt)>("synLaunchWithExternalEventsExt");Scope scope;auto rc=f(s,t,n,w,r,e,ne,flags);row("synLaunchWithExternalEventsExt",n,-1,-1,rc,scope.outer,nullptr,nullptr,s);return rc;}

extern "C" synStatus synEventRecord(synEventHandle e,synStreamHandle s){static auto f=real<decltype(&synEventRecord)>("synEventRecord");Scope scope;auto rc=f(e,s);row("synEventRecord",0,-1,-1,rc,scope.outer,nullptr,nullptr,s,e);return rc;}
extern "C" synStatus synStreamWaitEvent(const synStreamHandle s,synEventHandle e,const uint32_t flags){static auto f=real<decltype(&synStreamWaitEvent)>("synStreamWaitEvent");Scope scope;auto rc=f(s,e,flags);row("synStreamWaitEvent",flags,-1,-1,rc,scope.outer,nullptr,nullptr,s,e);return rc;}
extern "C" synStatus synStreamSynchronize(const synStreamHandle s){static auto f=real<decltype(&synStreamSynchronize)>("synStreamSynchronize");Scope scope;auto rc=f(s);row("synStreamSynchronize",0,-1,-1,rc,scope.outer,nullptr,nullptr,s);return rc;}
