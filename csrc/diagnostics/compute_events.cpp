// Transparent public-API observation, solely for bounded timing diagnostics.
// Does not record command lists, change bindings, or replay any operation.
#include <synapse_api.h>
#include <atomic>
#include <dlfcn.h>
#include <cstdlib>
static std::atomic<synStreamHandle> observed{nullptr};
static synStreamHandle measured=nullptr;
static synEventHandle begin_event=nullptr,end_event=nullptr;
extern "C" synStatus synLaunchWithExternalEventsExt(synStreamHandle stream,
    const synLaunchTensorInfoExt* tensors,uint32_t count,uint64_t workspace,
    synRecipeHandle recipe,synEventHandle* events,uint32_t n,uint32_t flags){
  static auto real=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(
      dlsym(RTLD_NEXT,"synLaunchWithExternalEventsExt"));
  if(!real)std::abort();
  auto result=real(stream,tensors,count,workspace,recipe,events,n,flags);
  if(result==synSuccess)observed.store(stream,std::memory_order_release);
  return result;
}
extern "C" int probe_bind_compute(unsigned device){
  if(measured)return 0;
  measured=observed.load(std::memory_order_acquire);if(!measured)return -1;
  int code=synEventCreate(&begin_event,device,EVENT_COLLECT_TIME);if(code)return code;
  return synEventCreate(&end_event,device,EVENT_COLLECT_TIME);
}
extern "C" int probe_begin(){return measured?synEventRecord(begin_event,measured):-1;}
extern "C" int probe_end(){return measured?synEventRecord(end_event,measured):-1;}
extern "C" int probe_finish(uint64_t* result){
  auto status=synEventQuery(begin_event);
  if(status!=synSuccess&&status!=synBusy)return status;
  result[0]=status==synBusy;
  status=synEventSynchronize(end_event);if(status!=synSuccess)return status;
  return synEventElapsedTime(result+1,begin_event,end_event);
}
