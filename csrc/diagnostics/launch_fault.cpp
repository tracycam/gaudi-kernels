// Explicit test-only failure before a selected launch is submitted to the SDK.
#include <synapse_api.h>
#include <atomic>
#include <dlfcn.h>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <chrono>
static std::atomic<unsigned> selected{0},calls{0};
static std::atomic<unsigned> pause_selected{0},pause_calls{0};
static std::mutex pause_mutex;
static std::condition_variable pause_cv;
static bool entered=false,released=false,expired=false;
extern "C" void launch_fault_arm(unsigned n){calls=0;selected=n;}
extern "C" unsigned launch_fault_calls(){return calls;}
extern "C" void launch_pause_arm(unsigned n){
  std::lock_guard<std::mutex> lock(pause_mutex);
  pause_calls=0;pause_selected=n;entered=released=expired=false;
}
extern "C" int launch_pause_wait_entered(unsigned ms){
  std::unique_lock<std::mutex> lock(pause_mutex);
  return pause_cv.wait_for(lock,std::chrono::milliseconds(ms),[]{return entered;});
}
extern "C" void launch_pause_release(){
  std::lock_guard<std::mutex> lock(pause_mutex);released=true;pause_cv.notify_all();
}
extern "C" int launch_pause_expired(){
  std::lock_guard<std::mutex> lock(pause_mutex);return expired;
}
extern "C" synStatus synLaunchWithExternalEventsExt(synStreamHandle s,
    const synLaunchTensorInfoExt* t,uint32_t n,uint64_t w,synRecipeHandle r,
    synEventHandle* e,uint32_t ne,uint32_t flags){
  static auto fn=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(dlsym(RTLD_NEXT,"synLaunchWithExternalEventsExt"));
  if(!fn)std::abort();
  auto pause=pause_selected.load();
  if(pause && ++pause_calls==pause){
    std::unique_lock<std::mutex> lock(pause_mutex);
    entered=true;pause_cv.notify_all();
    if(!pause_cv.wait_for(lock,std::chrono::seconds(2),[]{return released;})){
      expired=true;return synInvalidArgument;
    }
  }
  auto target=selected.load();
  if(target && ++calls==target)return synInvalidArgument;
  return fn(s,t,n,w,r,e,ne,flags);
}
