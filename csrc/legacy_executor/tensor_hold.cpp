// Loaded after torch and the Habana bridge. This library never enters LD_PRELOAD.
// The capture interposer passes opaque references from RecipeLauncher::Launch.
#include <ATen/core/ivalue.h>
#include <c10/core/Storage.h>
#include <dlfcn.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using Values = std::vector<std::shared_ptr<c10::IValue>>;
struct HoldState {
  std::mutex mutex;
  std::vector<c10::Storage> storages;
  std::vector<at::Tensor> tensors;
  std::set<const void*> seen;
  std::vector<std::string> records;
  uint64_t calls=0, tensor_refs=0;
};
// The container has process lifetime, but native_tensor_hold_release explicitly
// drains the plan and clears its leases while the HPU allocator is still alive.
static HoldState& state() { static auto* s=new HoldState; return *s; }

static void keep(const c10::IValue& value,const char* role,size_t index) {
  if(!value.isTensor())return;
  auto tensor=value.toTensor();
  if(!tensor.defined()||!tensor.has_storage())return;
  auto storage=tensor.storage();
  auto& s=state();++s.tensor_refs;
  const void* identity=storage.unsafeGetStorageImpl();
  if(s.seen.insert(identity).second){
    s.storages.push_back(storage);
    s.tensors.push_back(tensor);
  }
  char line[512];
  std::snprintf(line,sizeof line,
    "call=%llu role=%s index=%zu storage=%p base=%p offset_elements=%lld dtype=%d bytes=%zu shape=",
    (unsigned long long)s.calls,role,index,identity,storage.data_ptr().get(),
    (long long)tensor.storage_offset(),(int)tensor.scalar_type(),storage.nbytes());
  std::string record=line;
  for(auto n:tensor.sizes())record+=std::to_string(n)+",";
  record+=" strides=";
  for(auto n:tensor.strides())record+=std::to_string(n)+",";
  s.records.push_back(std::move(record));
}

static int capture(const void* inputs,const void* intermediate,const void* outputs,const void* dma) {
  try {
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);++s.calls;
    const auto& in=*static_cast<const c10::ArrayRef<c10::IValue>*>(inputs);
    const auto& mid=*static_cast<const std::shared_ptr<Values>*>(intermediate);
    const auto& out=*static_cast<const Values*>(outputs);
    const auto& transfers=*static_cast<const Values*>(dma);
    for(size_t i=0;i<in.size();++i)keep(in[i],"input",i);
    if(mid)for(size_t i=0;i<mid->size();++i)if((*mid)[i])keep(*(*mid)[i],"intermediate",i);
    for(size_t i=0;i<out.size();++i)if(out[i])keep(*out[i],"output",i);
    for(size_t i=0;i<transfers.size();++i)if(transfers[i])keep(*transfers[i],"dma",i);
    return 0;
  }catch(const std::exception& e){fprintf(stderr,"E1_TENSOR_HOLD_ERROR %s\n",e.what());return 1;}
}

extern "C" int e1_tensor_hold_install() {
  using Callback=int(*)(const void*,const void*,const void*,const void*);
  using Set=void(*)(Callback);
  auto set=reinterpret_cast<Set>(dlsym(RTLD_DEFAULT,"e1_set_hold_callback"));
  if(!set)return -1;
  set(capture);return 0;
}
extern "C" int e1_tensor_hold_stats(uint64_t* result,const char* path) {
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  result[0]=s.calls;result[1]=s.tensor_refs;result[2]=s.storages.size();
  if(path){
    FILE* file=fopen(path,"w");if(!file)return -1;
    for(const auto& line:s.records)fprintf(file,"%s\n",line.c_str());
    fclose(file);
  }
  return 0;
}

// The caller must drain/destroy the native plan first. Model-owned weights/KV
// remain alive; these leases only prevent captured intermediates being reused.
extern "C" int native_tensor_hold_release() {
  using Reset=int(*)();
  auto reset=reinterpret_cast<Reset>(dlsym(RTLD_DEFAULT,"native_plan_reset"));
  if(!reset)return -1;
  int code=reset();if(code)return code;
  auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  s.tensors.clear();s.storages.clear();s.seen.clear();s.records.clear();
  s.calls=0;s.tensor_refs=0;return 0;
}
