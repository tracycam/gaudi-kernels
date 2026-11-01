// Transparent, opt-in CPU API timing. No capture/replay or device profiler.
#include <synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
namespace {
constexpr unsigned N=7;
std::atomic<bool> enabled{false};
std::atomic<synStreamHandle> observed_compute_stream{nullptr};
synEventHandle feedback_gate_event=nullptr;
struct Metric { std::atomic<uint64_t> count{0},ns{0},outer_count{0},outer_ns{0},peak_ns{0},peak_index{0}; } metrics[N];
thread_local unsigned depth=0;
uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Scope {
  unsigned id; bool active,outer; uint64_t start;
  explicit Scope(unsigned i):id(i),active(enabled.load(std::memory_order_relaxed)),outer(depth++==0),start(active?now():0){}
  ~Scope(){--depth;if(active){auto elapsed=now()-start;auto& m=metrics[id];auto index=++m.count;m.ns+=elapsed;auto peak=m.peak_ns.load();while(elapsed>peak){if(m.peak_ns.compare_exchange_weak(peak,elapsed)){m.peak_index=index;break;}}if(outer){m.outer_count++;m.outer_ns+=elapsed;}}}
};
template<class T>T resolve(const char* name){
  auto p=dlsym(RTLD_NEXT,name);
  // HCCL may have been opened RTLD_LOCAL by the bridge.
  if(!p)for(const auto file:{"/usr/lib/habanalabs/libSynapse.so","/usr/lib/habanalabs/libhccl.so"}){
    if(auto h=dlopen(file,RTLD_NOW|RTLD_NOLOAD)){p=dlsym(h,name);dlclose(h);if(p)break;}
  }
  if(!p){std::fprintf(stderr,"API timing cannot resolve %s\n",name);std::abort();}
  return reinterpret_cast<T>(p);
}
}
extern "C" int api_timing_begin(){if(enabled.exchange(false))return -1;for(auto& m:metrics){m.count=0;m.ns=0;m.outer_count=0;m.outer_ns=0;m.peak_ns=0;m.peak_index=0;}enabled=true;return 0;}
extern "C" void api_timing_end(uint64_t* out){enabled=false;for(unsigned i=0;i<N;++i){out[4*i]=metrics[i].count;out[4*i+1]=metrics[i].ns;out[4*i+2]=metrics[i].outer_count;out[4*i+3]=metrics[i].outer_ns;}}
extern "C" void api_timing_peaks(uint64_t* out){for(unsigned i=0;i<N;++i)out[i]=metrics[i].peak_ns;}
extern "C" void api_timing_peak_indices(uint64_t* out){for(unsigned i=0;i<N;++i)out[i]=metrics[i].peak_index;}
extern "C" synStatus synLaunchWithExternalEvents(synStreamHandle s,const synLaunchTensorInfo* t,uint32_t n,uint64_t w,synRecipeHandle r,synEventHandle* e,uint32_t ne,uint32_t f){static auto fn=resolve<decltype(&synLaunchWithExternalEvents)>("synLaunchWithExternalEvents");Scope scope(0);return fn(s,t,n,w,r,e,ne,f);}
extern "C" synStatus synLaunchWithExternalEventsExt(synStreamHandle s,const synLaunchTensorInfoExt* t,uint32_t n,uint64_t w,synRecipeHandle r,synEventHandle* e,uint32_t ne,uint32_t f){static auto fn=resolve<decltype(&synLaunchWithExternalEventsExt)>("synLaunchWithExternalEventsExt");Scope scope(1);auto status=fn(s,t,n,w,r,e,ne,f);if(status==synSuccess)observed_compute_stream.store(s,std::memory_order_release);return status;}
extern "C" hcclResult_t hcclAllGather(const void* a,void* b,size_t n,hcclDataType_t d,hcclComm_t c,void* s){static auto fn=resolve<decltype(&hcclAllGather)>("hcclAllGather");Scope scope(2);return fn(a,b,n,d,c,s);}
extern "C" hcclResult_t hcclAllReduce(const void* a,void* b,size_t n,hcclDataType_t d,hcclRedOp_t op,hcclComm_t c,void* s){static auto fn=resolve<decltype(&hcclAllReduce)>("hcclAllReduce");Scope scope(3);return fn(a,b,n,d,op,c,s);}
extern "C" synStatus synEventRecord(synEventHandle e,synStreamHandle s){static auto fn=resolve<decltype(&synEventRecord)>("synEventRecord");Scope scope(4);return fn(e,s);}
extern "C" synStatus synStreamWaitEvent(synStreamHandle s,synEventHandle e,uint32_t f){static auto fn=resolve<decltype(&synStreamWaitEvent)>("synStreamWaitEvent");Scope scope(5);return fn(s,e,f);}
extern "C" synStatus synEventSynchronize(synEventHandle e){static auto fn=resolve<decltype(&synEventSynchronize)>("synEventSynchronize");Scope scope(6);return fn(e);}
// Explicit diagnostic-only completion witness. The caller owns the source
// Tensor and mapped host page until stream completion. No runtime ownership,
// replay, queue replacement, or production path is added by these helpers.
extern "C" int feedback_host_map(unsigned device,uint64_t bytes,void* pointer){
  return resolve<decltype(&synHostMap)>("synHostMap")(device,bytes,pointer);
}
extern "C" int feedback_host_unmap(unsigned device,void* pointer){
  return resolve<decltype(&synHostUnmap)>("synHostUnmap")(device,pointer);
}
extern "C" int feedback_copy(uint64_t source,uint64_t bytes,uint64_t destination){
  auto stream=observed_compute_stream.load(std::memory_order_acquire);
  if(!stream)return -1;
  return resolve<decltype(&synMemCopyAsync)>("synMemCopyAsync")(
    stream,source,bytes,destination,DRAM_TO_HOST);
}
extern "C" int feedback_gate_record(unsigned device){
  if(feedback_gate_event)return -1;
  auto stream=observed_compute_stream.load(std::memory_order_acquire);
  if(!stream)return -2;
  auto code=resolve<decltype(&synEventCreate)>("synEventCreate")(&feedback_gate_event,device,0);
  if(code!=synSuccess)return code;
  return resolve<decltype(&synEventRecord)>("synEventRecord")(feedback_gate_event,stream);
}
extern "C" int feedback_gate_query(){
  if(!feedback_gate_event)return -1;
  return resolve<decltype(&synEventQuery)>("synEventQuery")(feedback_gate_event);
}
extern "C" int feedback_gate_destroy(){
  if(!feedback_gate_event)return 0;
  auto code=resolve<decltype(&synEventDestroy)>("synEventDestroy")(feedback_gate_event);
  if(code==synSuccess)feedback_gate_event=nullptr;
  return code;
}
