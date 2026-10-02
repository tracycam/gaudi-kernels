// One-shot, same-process command recording; never execute an incomplete command list.
#include <synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <array>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <map>
#include "replay_schedule.hpp"

static void* resolve(const char* name) {
  void* p = dlsym(RTLD_NEXT, name);
  if (!p) { void* h = dlopen("/usr/lib/habanalabs/libSynapse.so", RTLD_NOW | RTLD_NOLOAD); if (h) p=dlsym(h,name); }
  if (!p) { fprintf(stderr,"E1 missing %s\n",name); abort(); }
  return p;
}
#define REAL(f) static auto real=reinterpret_cast<decltype(&f)>(resolve(#f))
struct Command {
  int kind=0,type=0,op=0; uint32_t flags=0;
  void* object=nullptr; void* src=nullptr; void* dst=nullptr;
  uint64_t count=0,workspace=0;
  std::vector<synLaunchTensorInfoExt> tensors;
  std::vector<std::string> names;
  std::string semantic_role;
  std::vector<synRetrievedLaunchTensorInfoExt> metadata;
  std::vector<bool> metadata_valid;
  std::vector<unsigned char> host_bytes;
  std::vector<uint64_t> multi_src,multi_dst,multi_size;
  std::vector<std::vector<unsigned char>> multi_host_bytes;
  struct Image { uint64_t address=0, bytes=0; std::string name; void* host=nullptr; uint64_t full_bytes=0; };
  std::vector<Image> before, after;
};
static std::mutex mutex_;
static std::vector<Command> commands;
static std::vector<size_t> compact_commands;
// 0: compact aggregate timing, 1: historical per-record timing,
// 2: compact API diagnostic, 3: compact API + device-event diagnostic.
static int replay_mode=0;
static std::atomic<bool> active{false};
// Explicit startup ABI. Never consult the environment in capture/replay.
static bool options_initialized=false;
static uint32_t option_flags=0;
static std::string backend_library="libhabana_pytorch_backend.so";
extern "C" int e1_configure_options(uint32_t flags,const char* backend) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(active.load() || !commands.empty() || (flags & ~uint32_t(127)))return -1;
  if(options_initialized)return -2;
  if(backend && !*backend)return -3;
  option_flags=flags;
  if(backend)backend_library=backend;
  options_initialized=true;
  return 0;
}
static bool option(uint32_t bit){return (option_flags & bit)!=0;}
extern "C" int e1_set_replay_mode(int mode) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(active.load()||!commands.empty()||mode<0||mode>3)return -1;
  replay_mode=mode;return 0;
}

static thread_local int depth=0;
static thread_local std::string semantic_role;
extern "C" void e1_set_semantic_role(const char* role) { semantic_role=role?role:""; }
struct Guard { Guard(){++depth;} ~Guard(){--depth;} };
static synStreamHandle serial_stream=nullptr;
static synDeviceId device=0;
static std::vector<void*> owned_pinned_buffers;
static int rejected=0,launches=0,collectives=0,memcopies=0,external_events=0,external_waits=0,syncs=0;
static std::set<void*> recorded_events;
static bool diagnostic=false;
static uint64_t snapshot_bytes=0;
static std::atomic<uint64_t> metadata_queries{0}, metadata_tensors{0}, metadata_images{0}, metadata_empty{0};
static std::map<void*,std::vector<synRetrievedLaunchTensorInfoExt>> recipe_metadata;
static std::atomic<uint64_t> metadata_omitted{0}, partial_outputs{0};
static constexpr uint64_t max_full_output=4ull*1024*1024;
static constexpr uint64_t max_snapshot_bytes=256ull*1024*1024;
static void reject(int reason);
static size_t element_bits(synDataType type) {
  if(type==syn_type_bf16||type==syn_type_fp16||type==syn_type_int16||type==syn_type_uint16||type==syn_type_ufp16)return 16;
  if(type==syn_type_single||type==syn_type_int32||type==syn_type_uint32||type==syn_type_tf32||type==syn_type_hb_float)return 32;
  if(type==syn_type_int64||type==syn_type_uint64)return 64;
  if(type==syn_type_packed_mxfp4||type==syn_type_packed_nf4||type==syn_type_int4||type==syn_type_uint4)return 4;
  if(type==syn_type_fixed||type==syn_type_uint8||type==syn_type_fp8_143||type==syn_type_fp8_152)return 8;
  return 0;
}
static void image(Command& c, uint64_t address, uint64_t bytes, std::string name,
                  synStreamHandle stream, bool after) {
  if(!diagnostic||!address||!bytes||(rejected&&rejected!=3&&rejected!=4&&rejected!=8&&rejected!=9))return;
  if(bytes>max_snapshot_bytes-snapshot_bytes){reject(20);return;}
  void* host=nullptr;
  synStatus status;
  {Guard g;status=synHostMalloc(device,bytes,0,&host);}
  if(status!=synSuccess){reject(21);return;}
  {Guard g;status=synMemCopyAsync(stream,address,bytes,(uint64_t)host,DRAM_TO_HOST);}
  if(status!=synSuccess){reject(22);return;}
  snapshot_bytes+=bytes;
  (after?c.after:c.before).push_back({address,bytes,std::move(name),host});
}
static void launch_images(Command& c, synStreamHandle stream, bool after) {
  if(!diagnostic)return;
  // Only the compiled recipe inventory defines a live launch tensor. The
  // bridge also submits names eliminated by compilation; by-name queries can
  // return success with zero metadata for these, which is not a real dtype.
  if(c.metadata.empty()){
    auto it=recipe_metadata.find(c.object);
    if(it==recipe_metadata.end()){
      uint32_t count=0; synStatus status;
      std::vector<uint64_t> ids;
      std::vector<synRetrievedLaunchTensorInfoExt> inventory;
      {
        Guard g;
        auto amount=reinterpret_cast<decltype(&synTensorRetrieveLaunchAmount)>(resolve("synTensorRetrieveLaunchAmount"));
        auto getids=reinterpret_cast<decltype(&synTensorRetrieveLaunchIds)>(resolve("synTensorRetrieveLaunchIds"));
        auto query=reinterpret_cast<decltype(&synTensorRetrieveLaunchInfoByIdExt)>(resolve("synTensorRetrieveLaunchInfoByIdExt"));
        status=amount((synRecipeHandle)c.object,&count);
        if(status==synSuccess){ids.resize(count);status=getids((synRecipeHandle)c.object,ids.data(),count);}
        if(status==synSuccess){
          inventory.resize(count);
          for(size_t i=0;i<count;++i)inventory[i].tensorId=ids[i];
          status=query((synRecipeHandle)c.object,count,inventory.data());
        }
      }
      if(status!=synSuccess||!count){reject(23);return;}
      metadata_queries.fetch_add(1);
      it=recipe_metadata.emplace(c.object,std::move(inventory)).first;
    }
    c.metadata.resize(c.tensors.size());c.metadata_valid.resize(c.tensors.size(),false);
    std::vector<bool> found(it->second.size(),false);
    for(size_t i=0;i<c.tensors.size();++i){
      for(size_t j=0;j<it->second.size();++j){const auto& m=it->second[j];
        bool match=(c.flags&SYN_FLAGS_TENSOR_NAME)? c.names[i]==m.tensorName : c.tensors[i].tensorId==m.tensorId;
        if(match){c.metadata[i]=m;c.metadata_valid[i]=true;found[j]=true;break;}
      }
      if(!c.metadata_valid[i])metadata_omitted.fetch_add(1);
    }
    for(size_t j=0;j<found.size();++j)if(!found[j]){
      fprintf(stderr,"E1_MISSING_BINDING recipe=%p name=%s id=%llu\n",c.object,it->second[j].tensorName,(unsigned long long)it->second[j].tensorId);
      reject(27);return;
    }
  }
  const auto& infos=c.metadata;
  for(size_t i=0;i<infos.size();++i){
    if(!c.metadata_valid[i] || (!infos[i].isInput)!=after || c.tensors[i].tensorType!=DATA_TENSOR)continue;
    metadata_tensors.fetch_add(1);
    auto width=element_bits(infos[i].tensorDataType);
    if(!width){fprintf(stderr,"E1_UNSUPPORTED_DTYPE recipe=%p name=%s id=%llu dtype=%d input=%d\n",
                      c.object,c.names[i].c_str(),(unsigned long long)c.tensors[i].tensorId,
                      (int)infos[i].tensorDataType,(int)infos[i].isInput);reject(24);return;}
    uint64_t bits=width;
    for(uint32_t dim=0;dim<infos[i].tensorDims;++dim){
      auto size=c.tensors[i].tensorSize[dim];
      if(!size){bits=0;break;}
      if(!after && bits>(256*8)/size){bits=256*8;break;}
      if(after && bits>std::numeric_limits<uint64_t>::max()/size){reject(25);return;}
      bits*=size;
    }
    if(!bits){metadata_empty.fetch_add(1);continue;}
    uint64_t bytes=(bits+7)/8;
    // A full output is required for bitwise comparison; input previews are
    // explicitly bounded and only identify the first differing input.
    if(!after && bytes>256)bytes=256;
    uint64_t full_bytes=bytes;
    if(after && bytes>max_full_output){bytes=65536;partial_outputs.fetch_add(1);}
    image(c,c.tensors[i].pTensorAddress,bytes,c.names[i],stream,after);
    auto& images=after?c.after:c.before;
    if(!images.empty())images.back().full_bytes=full_bytes;
    metadata_images.fetch_add(1);
  }
}
static size_t add(Command c) {
  if (!active.load() || depth) return SIZE_MAX;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active.load()) return SIZE_MAX;
  size_t index=commands.size();commands.push_back(std::move(c));return index;
}
static void reject(int reason) {
  if (!active.load() || depth) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!rejected || reason>=20) rejected=reason;
}
using HoldCallback=int(*)(const void*,const void*,const void*,const void*);
static std::atomic<HoldCallback> hold_callback{nullptr};
extern "C" void e1_set_hold_callback(HoldCallback callback) { hold_callback.store(callback); }
// ABI verified against the installed bridge symbol and its PLT relocation.
// Opaque references keep torch out of this preload library's dependencies.
#define RECIPE_LAUNCH_SYMBOL "_ZN6habana14RecipeLauncher6LaunchEmRKN3c108ArrayRefINS1_6IValueEEERSt10shared_ptrISt6vectorIS7_IS3_ESaIS9_EEERKSB_RS8_I19synLaunchTensorInfoSaISG_EERS8_ImSaImEESF_"
extern "C" void e1_bridge_launch(void*,uint64_t,const void*,void*,const void*,void*,void*,const void*) asm(RECIPE_LAUNCH_SYMBOL);
extern "C" void e1_bridge_launch(void* self,uint64_t stream,const void* inputs,void* mid,
    const void* outputs,void* bindings,void* events,const void* dma) {
  using Launch=void(*)(void*,uint64_t,const void*,void*,const void*,void*,void*,const void*);
  static auto real=[](){
    void* symbol=dlsym(RTLD_NEXT,RECIPE_LAUNCH_SYMBOL);
    if(!symbol){
      const char* path=backend_library.c_str();
      void* library=dlopen(path?path:"libhabana_pytorch_backend.so",RTLD_NOW|RTLD_NOLOAD);
      if(library)symbol=dlsym(library,RECIPE_LAUNCH_SYMBOL);
    }
    if(!symbol){fprintf(stderr,"E1 missing RecipeLauncher::Launch\n");abort();}
    return reinterpret_cast<Launch>(symbol);
  }();
  auto callback=hold_callback.load();
  if(active.load() && !depth && callback && callback(inputs,mid,outputs,dma))reject(40);
  real(self,stream,inputs,mid,outputs,bindings,events,dma);
}
extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt* ts,
    const uint32_t n,uint64_t workspace,const synRecipeHandle recipe,synEventHandle* events,const uint32_t ne,uint32_t flags) {
  REAL(synLaunchWithExternalEventsExt);
  if (active.load() && !depth) {
    if (ne) { ++external_events; reject(2); }
    Command c; c.kind=1; c.object=(void*)recipe; c.workspace=workspace; c.flags=flags;
    c.tensors.assign(ts,ts+n);
    for (uint32_t i=0;i<n;++i) c.names.emplace_back(ts[i].tensorName ? ts[i].tensorName : "");
    launch_images(c,s,false);
    size_t index=add(c);
    synStatus status;
    { Guard g;status=real(s,ts,n,workspace,recipe,events,ne,flags); }
    launch_images(c,s,true);
    if(index!=SIZE_MAX){std::lock_guard<std::mutex> lock(mutex_);commands[index].after=std::move(c.after);}
    return status;
  }
  Guard g; return real(s,ts,n,workspace,recipe,events,ne,flags);
}
extern "C" hcclResult_t hcclAllReduce(const void* src,void* dst,size_t count,hcclDataType_t type,
                                        hcclRedOp_t op,hcclComm_t comm,void* stream) {
  REAL(hcclAllReduce); Command c; c.kind=2;c.src=(void*)src;c.dst=dst;c.count=count;c.type=type;c.op=op;c.object=(void*)comm;
  if(active.load()&&!depth&&diagnostic){
    size_t width=(type==hcclFloat64||type==hcclInt64||type==hcclUint64)?8:
      (type==hcclFloat32||type==hcclInt32||type==hcclUint32)?4:
      (type==hcclFloat16||type==hcclBfloat16)?2:1;
    image(c,(uint64_t)src,std::min<uint64_t>(count*width,256),"allreduce-input",(synStreamHandle)stream,false);
  }
  size_t index=add(c);
  hcclResult_t status;
  {Guard g;status=real(src,dst,count,type,op,comm,stream);}
  if(active.load()&&!depth && diagnostic){
    size_t width=(type==hcclFloat64||type==hcclInt64||type==hcclUint64)?8:
      (type==hcclFloat32||type==hcclInt32||type==hcclUint32)?4:
      (type==hcclFloat16||type==hcclBfloat16)?2:1;
    image(c,(uint64_t)dst,count*width,"allreduce",(synStreamHandle)stream,true);
  }
  if(index!=SIZE_MAX){std::lock_guard<std::mutex> lock(mutex_);commands[index].after=std::move(c.after);}
  return status;
}
extern "C" hcclResult_t hcclAllGather(const void* src,void* dst,size_t count,hcclDataType_t type,hcclComm_t comm,void* stream) {
  REAL(hcclAllGather); Command c;c.kind=3;c.src=(void*)src;c.dst=dst;c.count=count;c.type=type;c.object=(void*)comm;
  if(active.load()&&!depth&&diagnostic){
    size_t width=(type==hcclFloat64||type==hcclInt64||type==hcclUint64)?8:
      (type==hcclFloat32||type==hcclInt32||type==hcclUint32)?4:
      (type==hcclFloat16||type==hcclBfloat16)?2:1;
    image(c,(uint64_t)src,std::min<uint64_t>(count*width,256),"allgather-input",(synStreamHandle)stream,false);
  }
  size_t index=add(c);
  hcclResult_t status;
  {Guard g;status=real(src,dst,count,type,comm,stream);}
  // Gather output is rank-count times the per-rank input; infer rank count
  // from the communicator instead of trusting a hardcoded TP size.
  if(active.load()&&!depth&&diagnostic){int ranks=0;auto rc=hcclCommCount(comm,&ranks);
    size_t width=(type==hcclFloat64||type==hcclInt64||type==hcclUint64)?8:
      (type==hcclFloat32||type==hcclInt32||type==hcclUint32)?4:
      (type==hcclFloat16||type==hcclBfloat16)?2:1;
    if(rc||ranks<=0)reject(26);else image(c,(uint64_t)dst,count*ranks*width,"allgather",(synStreamHandle)stream,true);
  }
  if(index!=SIZE_MAX){std::lock_guard<std::mutex> lock(mutex_);commands[index].after=std::move(c.after);}
  return status;
}
extern "C" synStatus synMemCopyAsync(const synStreamHandle s,uint64_t src,uint64_t n,uint64_t dst,const synDmaDir direction) {
  REAL(synMemCopyAsync);
  if(active.load()&&!depth){
    ++memcopies;Command c;c.kind=6;c.src=(void*)src;c.dst=(void*)dst;c.count=n;c.type=(int)direction;c.semantic_role=semantic_role;
    if(direction==HOST_TO_DRAM && n){c.host_bytes.resize(n);std::memcpy(c.host_bytes.data(),(void*)src,n);}
    size_t index=add(c);reject(3);
    synStatus status;
    {Guard g;status=real(s,src,n,dst,direction);}
    if(diagnostic&&status==synSuccess&&n){
      if(direction==HOST_TO_DRAM)image(c,dst,n,"h2d",s,true);
      else if(direction==DRAM_TO_HOST)image(c,src,n,"d2h-source",s,false);
      if(index!=SIZE_MAX){std::lock_guard<std::mutex> lock(mutex_);
        commands[index].before=std::move(c.before);commands[index].after=std::move(c.after);}
    }
    return status;
  }
  Guard g;return real(s,src,n,dst,direction);
}
extern "C" synStatus synMemCopyAsyncMultiple(const synStreamHandle s,const uint64_t* src,const uint64_t* n,
    const uint64_t* dst,const synDmaDir direction,const uint64_t count) {
  REAL(synMemCopyAsyncMultiple);
  if(active.load()&&!depth){
    memcopies+=count;
    Command c;c.kind=7;c.type=(int)direction;c.semantic_role=semantic_role;
    c.multi_src.assign(src,src+count);c.multi_size.assign(n,n+count);c.multi_dst.assign(dst,dst+count);
    c.multi_host_bytes.resize(count);
    for(uint64_t i=0;i<count;++i)if(direction==HOST_TO_DRAM && n[i]){
      c.multi_host_bytes[i].resize(n[i]);std::memcpy(c.multi_host_bytes[i].data(),(void*)src[i],n[i]);
    }
    add(std::move(c));
    reject(4);
  }
  Guard g;return real(s,src,n,dst,direction,count);
}
extern "C" synStatus synEventRecord(synEventHandle e,const synStreamHandle s) {
  REAL(synEventRecord);
  if(active.load()&&!depth){
    Command c;c.kind=4;c.object=(void*)e;add(std::move(c));
    std::lock_guard<std::mutex> lock(mutex_);recorded_events.insert((void*)e);
  }
  Guard g;return real(e,s);
}
extern "C" synStatus synStreamWaitEvent(const synStreamHandle s,synEventHandle e,const uint32_t f) {
  REAL(synStreamWaitEvent);
  if(active.load()&&!depth){
    Command c;c.kind=5;c.object=(void*)e;c.flags=f;add(std::move(c));
    std::lock_guard<std::mutex> lock(mutex_);
    if(!recorded_events.count((void*)e)){++external_waits;if(!rejected)rejected=7;}
  }
  Guard g;return real(s,e,f);
}
extern "C" synStatus synStreamSynchronize(const synStreamHandle s) {
  REAL(synStreamSynchronize);
  if(active.load()&&!depth){++syncs;Command c;c.kind=8;c.object=(void*)s;add(std::move(c));reject(8);}
  Guard g;return real(s);
}
extern "C" synStatus synEventSynchronize(const synEventHandle e) {
  REAL(synEventSynchronize);
  if(active.load()&&!depth){++syncs;Command c;c.kind=9;c.object=(void*)e;add(std::move(c));reject(9);}
  Guard g;return real(e);
}
extern "C" int e1_begin(int dev) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!options_initialized) return -60;
  if (active.load()||!commands.empty()) return -1;
  device=dev;
  diagnostic=option(1);
  snapshot_bytes=0;
  rejected=launches=collectives=memcopies=external_events=external_waits=syncs=0;
  recorded_events.clear();
  active.store(true);return 0;
}
extern "C" int e1_is_active() { return active.load() ? 1 : 0; }
extern "C" int e1_pause() {
  std::lock_guard<std::mutex> lock(mutex_);active.store(false);return rejected;
}
extern "C" int e1_end(int dev, const char* filename) {
  std::lock_guard<std::mutex> lock(mutex_);active.store(false);
  FILE* out=fopen(filename,"w");
  for (auto& c:commands) {
    if(c.kind==1) { ++launches;
      for(size_t i=0;i<c.tensors.size();++i) c.tensors[i].tensorName=c.names[i].c_str();
    } else if(c.kind==2||c.kind==3) ++collectives;
  }
  if(out) {
    fprintf(out,"commands=%zu launches=%d collectives=%d memcpy=%d ext_events=%d external_waits=%d syncs=%d rejected=%d\n",
            commands.size(),launches,collectives,memcopies,external_events,external_waits,syncs,rejected);
    for(size_t i=0;i<commands.size();++i){ auto& c=commands[i];
      fprintf(out,"%zu kind=%d type=%d recipe_or_comm=%p src=%p dst=%p count=%llu workspace=%llu tensors=%zu multi=%zu role=%s\n",i,c.kind,c.type,c.object,c.src,c.dst,(unsigned long long)c.count,(unsigned long long)c.workspace,c.tensors.size(),c.multi_src.size(),c.semantic_role.c_str());
      if(c.kind==7)for(size_t j=0;j<c.multi_src.size();++j)
        fprintf(out,"  memcpy[%zu] src=%llu dst=%llu bytes=%llu\n",j,
                (unsigned long long)c.multi_src[j],(unsigned long long)c.multi_dst[j],
                (unsigned long long)c.multi_size[j]);
      for(size_t j=0;j<c.tensors.size();++j){
        fprintf(out,"  %s addr=%llu id=%llu type=%d dims0=%llu",c.names[j].c_str(),
                (unsigned long long)c.tensors[j].pTensorAddress,(unsigned long long)c.tensors[j].tensorId,
                (int)c.tensors[j].tensorType,(unsigned long long)c.tensors[j].tensorSize[0]);
        if(j<c.metadata.size()){
          const auto&m=c.metadata[j];fprintf(out," live=%d dtype=%d input=%d section=%u offset=%llu shape=",(int)c.metadata_valid[j],(int)m.tensorDataType,(int)m.isInput,m.tensorSectionId,(unsigned long long)m.tensorOffsetInSection);
          for(unsigned d=0;d<m.tensorDims;++d)fprintf(out,"%s%llu",d?",":"",(unsigned long long)c.tensors[j].tensorSize[d]);
        }
        fprintf(out,"\n");
      }
    }
    fclose(out);
  }
  if(external_waits||external_events)return rejected?-rejected:-12;
  bool allow_sync=option(2);
  bool allow_memcpy=option(4);
  // The fixed single stream replaces bridge waits only after repeated numeric
  // replay proves the producer-consumer chain; no host-visible branch occurs
  // until the final sampled ID, which is checked separately.
  if(syncs&&!allow_sync)return -8;
  if(memcopies&&!allow_memcpy)return -3;
  if(rejected && rejected!=3 && rejected!=4 && rejected!=8 && rejected!=9)return -rejected;
  bool single_fixture=option(8);
  if(!launches||(!collectives&&!single_fixture))return -5;
  device=dev;
  for(auto& c:commands)if(c.kind==6||c.kind==7){
    const auto direction=(synDmaDir)c.type;
    auto pin=[&](uint64_t& pointer,const uint64_t bytes,const std::vector<unsigned char>& saved){
      void* buffer=nullptr;
      auto status=synHostMalloc(device,bytes,0,&buffer);
      if(status!=synSuccess)return (int)status;
      owned_pinned_buffers.push_back(buffer);
      if(direction==HOST_TO_DRAM)std::memcpy(buffer,saved.data(),bytes);
      pointer=(uint64_t)buffer;
      return 0;
    };
    if(direction!=HOST_TO_DRAM&&direction!=DRAM_TO_HOST)continue;
    if(c.kind==6){
      if(!c.count)continue;
      uint64_t pointer=0;int status=pin(pointer,c.count,c.host_bytes);if(status)return -10;
      if(direction==HOST_TO_DRAM)c.src=(void*)pointer;else c.dst=(void*)pointer;
    }else for(size_t i=0;i<c.multi_src.size();++i){
      if(!c.multi_size[i])continue;
      uint64_t pointer=0;int status=pin(pointer,c.multi_size[i],c.multi_host_bytes[i]);if(status)return -11;
      if(direction==HOST_TO_DRAM)c.multi_src[i]=pointer;else c.multi_dst[i]=pointer;
    }
  }
  rejected=0;
  compact_commands=replay_indices(commands);
  auto create=reinterpret_cast<decltype(&synStreamCreateGeneric)>(resolve("synStreamCreateGeneric"));
  if(create(&serial_stream,device,0)!=synSuccess)return -6;
  return 0;
}
extern "C" int e1_stats(uint64_t* out) {
  out[0]=commands.size();out[1]=launches;out[2]=collectives;out[3]=memcopies;out[4]=external_events;
  return rejected;
}
extern "C" int e1_metadata_stats(uint64_t* out) {
  out[0]=metadata_queries.load();out[1]=metadata_tensors.load();
  out[2]=metadata_images.load();out[3]=metadata_empty.load();out[4]=snapshot_bytes;
  return rejected;
}
extern "C" int e1_coverage_stats(uint64_t* out) {
  out[0]=metadata_omitted.load();out[1]=partial_outputs.load();return 0;
}
// Perturb the actual model-supplied previous-token scalar. Its physical address
// is obtained before capture, while the model still owns that tensor. Require
// an observed int32 scalar read and reject any recorded overwrite of it.
extern "C" int e1_perturb_live_input(uint64_t address) {
  if(!serial_stream||rejected)return -1;
  int first=-1;
  for(size_t i=0;i<commands.size();++i){const auto& c=commands[i];
    if(c.kind==6 && c.type==HOST_TO_DRAM && address>=(uint64_t)c.dst && address<(uint64_t)c.dst+c.count)return -2;
    if(c.kind!=1)continue;
    for(size_t j=0;j<c.tensors.size();++j){
      if(j>=c.metadata_valid.size()||!c.metadata_valid[j]||c.tensors[j].pTensorAddress!=address)continue;
      const auto& m=c.metadata[j];
      if(!m.isInput)return -3;
      uint64_t elements=1;
      for(unsigned d=0;d<m.tensorDims;++d)elements*=c.tensors[j].tensorSize[d];
      if(m.tensorDataType!=syn_type_int32||elements!=1)return -4;
      if(first<0)first=(int)i;
    }
  }
  if(first<0)return -5;
  static void* host=nullptr;
  static uint64_t saved_address=0;
  static int32_t original=0;
  static bool changed=false;
  Guard guard;
  if(!host&&synHostMalloc(device,4,0,&host)!=synSuccess)return -6;
  if(!changed){
    if(synMemCopyAsync(serial_stream,address,4,(uint64_t)host,DRAM_TO_HOST)!=synSuccess || synStreamSynchronize(serial_stream)!=synSuccess)return -7;
    std::memcpy(&original,host,4);saved_address=address;
    int32_t replacement=original^1;std::memcpy(host,&replacement,4);
  }else{
    if(saved_address!=address)return -8;
    std::memcpy(host,&original,4);
  }
  if(synMemCopyAsync(serial_stream,(uint64_t)host,4,address,HOST_TO_DRAM)!=synSuccess || synStreamSynchronize(serial_stream)!=synSuccess)return -9;
  changed=!changed;return first;
}
// Flip one byte in the owned H2D source that feeds a live model input.
// Calling twice restores the captured token, without touching allocator state.
extern "C" int e1_perturb_input(uint64_t address) {
  for(size_t i=0;i<commands.size();++i){auto& c=commands[i];
    if(c.type!=HOST_TO_DRAM)continue;
    if(c.kind==6){
      uint64_t dst=(uint64_t)c.dst;
      if(address>=dst && address<dst+c.count){((unsigned char*)c.src)[address-dst]^=1;return (int)i;}
    }else if(c.kind==7)for(size_t j=0;j<c.multi_dst.size();++j){
      uint64_t dst=c.multi_dst[j];
      if(address>=dst && address<dst+c.multi_size[j]){
        ((unsigned char*)c.multi_src[j])[address-dst]^=1;return (int)i;
      }
    }
  }
  // A view may not point to its staging allocation. Probe the first small
  // H2D input only if the exact address is absent; the output-change gate
  // rejects a candidate that does not actually feed the token computation.
  for(size_t i=0;i<commands.size();++i){auto& c=commands[i];
    if(c.kind==6 && c.type==HOST_TO_DRAM && c.count==4){((unsigned char*)c.src)[0]^=1;return (int)i;}
  }
  return -1;
}
extern "C" int e1_sampled_id(int32_t* result) {
  for(auto it=commands.rbegin();it!=commands.rend();++it)if(it->kind==6 && it->type==DRAM_TO_HOST && it->count==4){
    std::memcpy(result,it->dst,4);return 0;
  }
  return -1;
}
extern "C" int e1_diagnose(const char* filename) {
  if(!diagnostic||!serial_stream||rejected)return -1;
  Guard guard;
  auto launch=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(resolve("synLaunchWithExternalEventsExt"));
  auto ar=reinterpret_cast<decltype(&hcclAllReduce)>(resolve("hcclAllReduce"));
  auto ag=reinterpret_cast<decltype(&hcclAllGather)>(resolve("hcclAllGather"));
  auto copy=reinterpret_cast<decltype(&synMemCopyAsync)>(resolve("synMemCopyAsync"));
  auto copies=reinterpret_cast<decltype(&synMemCopyAsyncMultiple)>(resolve("synMemCopyAsyncMultiple"));
  uint64_t largest=1;
  for(auto& c:commands)for(auto& v:c.before)largest=std::max(largest,v.bytes);
  for(auto& c:commands)for(auto& v:c.after)largest=std::max(largest,v.bytes);
  void* scratch=nullptr;
  if(synHostMalloc(device,largest,0,&scratch)!=synSuccess)return -2;
  FILE* out=fopen(filename,"w");if(!out)return -3;
  fprintf(out,"coverage=full_outputs_up_to_4MiB_larger_outputs_first_64KiB input_previews_first_256B omitted_nonrecipe_bindings=%llu partial_outputs=%llu\n",(unsigned long long)metadata_omitted.load(),(unsigned long long)partial_outputs.load());
  std::vector<bool> equal(commands.size(),true);
  int first=-1,first_input=-1,first_output=-1,code=0;
  auto dump=[&](const std::string& suffix,const void* p,uint64_t n){
    FILE* f=fopen((std::string(filename)+suffix).c_str(),"wb");if(f){fwrite(p,1,n,f);fclose(f);}
  };
  for(size_t i=0;i<commands.size();++i){
    auto& c=commands[i];
    // Every snapshot copy is ordered on the serial stream, then synchronized.
    for(size_t j=0;j<c.before.size();++j){auto& v=c.before[j];
      code=copy(serial_stream,v.address,v.bytes,(uint64_t)scratch,DRAM_TO_HOST);
      if(code || (code=synStreamSynchronize(serial_stream)))goto done;
      if(std::memcmp(scratch,v.host,v.bytes)){
        if(first_input<0)first_input=j;
        fprintf(out,"input_difference index=%zu input=%zu name=%s address=%llu bytes=%llu\n",i,j,v.name.c_str(),(unsigned long long)v.address,(unsigned long long)v.bytes);
        if(first<0){
          std::string tag=".command"+std::to_string(i)+".input"+std::to_string(j);
          dump(tag+".expected",v.host,v.bytes);dump(tag+".actual",scratch,v.bytes);
        }
      }
    }
    if(c.kind==1)code=launch(serial_stream,c.tensors.data(),c.tensors.size(),c.workspace,
                             (synRecipeHandle)c.object,nullptr,0,c.flags);
    else if(c.kind==2)code=ar(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclRedOp_t)c.op,
                              (hcclComm_t)c.object,serial_stream);
    else if(c.kind==3)code=ag(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclComm_t)c.object,serial_stream);
    else if(c.kind==6)code=copy(serial_stream,(uint64_t)c.src,c.count,(uint64_t)c.dst,(synDmaDir)c.type);
    else if(c.kind==7)code=copies(serial_stream,c.multi_src.data(),c.multi_size.data(),
                                 c.multi_dst.data(),(synDmaDir)c.type,c.multi_src.size());
    if(code || (code=synStreamSynchronize(serial_stream)))goto done;
    int differing=-1;
    for(size_t j=0;j<c.after.size();++j){auto& v=c.after[j];
      code=copy(serial_stream,v.address,v.bytes,(uint64_t)scratch,DRAM_TO_HOST);
      if(code || (code=synStreamSynchronize(serial_stream)))goto done;
      if(std::memcmp(scratch,v.host,v.bytes)){
        equal[i]=false;if(differing<0)differing=j;
        if(first<0){
          std::string tag=".command"+std::to_string(i)+".output"+std::to_string(j);
          dump(tag+".expected",v.host,v.bytes);dump(tag+".actual",scratch,v.bytes);
        }
      }
    }
    if(!equal[i]&&first<0){
      first=i;first_output=differing;
      auto& v=c.after[differing];
      fprintf(out,"first_output_coverage observed_bytes=%llu full_bytes=%llu full=%d\n",(unsigned long long)v.bytes,(unsigned long long)(v.full_bytes?v.full_bytes:v.bytes),(int)(!v.full_bytes||v.full_bytes==v.bytes));
    }
    fprintf(out,"index=%zu kind=%d recipe=%p output_equal=%d input_preview_diff=%d output_diff=%d\n",
            i,c.kind,c.object,(int)equal[i],first_input,differing);
    first_input=-1;
  }
  {
    // Binary search the all-prefix-equal predicate from the complete trace.
    std::vector<bool> prefix(equal.size()+1,true);
    for(size_t i=0;i<equal.size();++i)prefix[i+1]=prefix[i]&&equal[i];
    size_t lo=0,hi=equal.size();
    while(lo<hi){size_t mid=lo+(hi-lo)/2;if(prefix[mid+1])lo=mid+1;else hi=mid;}
    fprintf(out,"first_observed_divergent=%d binary_search=%zu kind=%d recipe=%p output=%s bytes=%llu\n",
            first,lo,first<0?-1:commands[first].kind,first<0?nullptr:commands[first].object,
            first<0||first_output<0?"-":commands[first].after[first_output].name.c_str(),
            (unsigned long long)(first<0||first_output<0?0:commands[first].after[first_output].bytes));
  }
done:
  if(code)fprintf(out,"diagnostic_error=%d\n",code);
  fclose(out);
  return code?-10000-code:first+1;
}
static int enqueue(unsigned times,uint64_t* host_ns,uint64_t* launch_ns,uint64_t* hccl_ns) {
  static auto launch=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(resolve("synLaunchWithExternalEventsExt"));
  static auto ar=reinterpret_cast<decltype(&hcclAllReduce)>(resolve("hcclAllReduce"));
  static auto ag=reinterpret_cast<decltype(&hcclAllGather)>(resolve("hcclAllGather"));
  static auto copy=reinterpret_cast<decltype(&synMemCopyAsync)>(resolve("synMemCopyAsync"));
  static auto copies=reinterpret_cast<decltype(&synMemCopyAsyncMultiple)>(resolve("synMemCopyAsyncMultiple"));
  Guard guard;
  auto clock=[](){return std::chrono::steady_clock::now();};
  auto start=clock();uint64_t l=0,h=0;
  const bool detailed=replay_mode!=0;
  const size_t count=replay_mode==1?commands.size():compact_commands.size();
  for(unsigned rep=0;rep<times;++rep)for(size_t i=0;i<count;++i){
    auto& c=commands[replay_mode==1?i:compact_commands[i]];
    auto t=detailed?clock():std::chrono::steady_clock::time_point{};int status=0;
    if(c.kind==1) status=launch(serial_stream,c.tensors.data(),c.tensors.size(),c.workspace,
                                (synRecipeHandle)c.object,nullptr,0,c.flags);
    else if(c.kind==2) status=ar(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclRedOp_t)c.op,
                                  (hcclComm_t)c.object,serial_stream);
    else if(c.kind==3) status=ag(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclComm_t)c.object,serial_stream);
    else if(c.kind==6) {
      // Diagnostic only: distinguish stale host staging from the model/collective replay.
      if(c.type!=HOST_TO_DRAM || !option(16))
        status=copy(serial_stream,(uint64_t)c.src,c.count,(uint64_t)c.dst,(synDmaDir)c.type);
    }
    else if(c.kind==7) status=copies(serial_stream,c.multi_src.data(),c.multi_size.data(),
                                     c.multi_dst.data(),(synDmaDir)c.type,c.multi_src.size());
    auto elapsed=detailed?std::chrono::duration_cast<std::chrono::nanoseconds>(clock()-t).count():0;
    if(c.kind==1)l+=elapsed;
    else if(c.kind==2||c.kind==3)h+=elapsed;
    if(status)return status;
  }
  *host_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(clock()-start).count();
  *launch_ns=l;*hccl_ns=h;
  return 0;
}
extern "C" uint32_t e1_host_buffer_addresses(uint64_t* out,uint32_t capacity) {
  for(uint32_t i=0;i<std::min<size_t>(capacity,owned_pinned_buffers.size());++i)
    out[i]=(uint64_t)owned_pinned_buffers[i];
  return owned_pinned_buffers.size();
}
extern "C" uint32_t e1_submission_count(){return compact_commands.size();}
extern "C" int e1_replay(unsigned times,uint64_t* host_ns,uint64_t* launch_ns,uint64_t* hccl_ns) {
  if(!serial_stream||rejected)return -1;
  auto sync=reinterpret_cast<decltype(&synStreamSynchronize)>(resolve("synStreamSynchronize"));
  int code=enqueue(times,host_ns,launch_ns,hccl_ns);
  return code?code:sync(serial_stream);
}

#include "e1_rollout.inc"

struct Timing {
  synGraphHandle graph{}; synRecipeHandle recipe{}; synTensor tensor[2]{};
  synSectionHandle section[2]{};synLaunchTensorInfo binding[2]{};
  uint64_t workspace{},workspace_size{},input{},output{};
  void* host=nullptr;
  synEventHandle gate_begin{},begin{},end{};
};
static Timing timing;
static uint64_t timing_storage=0,timing_storage_bytes=0;
extern "C" int e1_timing_set_storage(uint64_t address,uint64_t bytes) {
  if(timing.recipe||!address||bytes<768)return -1;
  timing_storage=address;timing_storage_bytes=bytes;return 0;
}
extern "C" int e1_timing_init() {
  if(!serial_stream||!commands.size())return -1;
#define CK(s) do { auto result=(s);if(result!=synSuccess){fprintf(stderr,"E1_TIMING_INIT line=%d status=%d\n",__LINE__,(int)result);return -__LINE__;} } while(0)
  CK(synGraphCreate(&timing.graph,synDeviceGaudi2));
  const char* names[]={"input","output"};
  for(int i=0;i<2;++i){
    CK(synSectionCreate(&timing.section[i],0,timing.graph));
    CK(synSectionSetPersistent(timing.section[i],true));
    synTensorDescriptor desc{};desc.m_name=names[i];desc.m_dataType=syn_type_float;desc.m_dims=1;
    desc.m_sizes[0]=desc.m_minSizes[0]=64;
    CK(synTensorCreate(&timing.tensor[i],&desc,timing.section[i],0));
    timing.binding[i].tensorName=names[i];timing.binding[i].tensorType=DATA_TENSOR;
  }
  struct Params {float scale,bias;int chunks;} params{1,0,-150000000};
  // Kernel interprets negative chunks as the bounded delay loop count.
  CK(synNodeCreate(timing.graph,timing.tensor,timing.tensor+1,1,1,&params,sizeof params,
                   "rf_affine_f32","e1_gate",nullptr,nullptr));
  CK(synGraphCompile(&timing.recipe,timing.graph,"e1_device_gate",nullptr));
  CK(synWorkspaceGetSize(&timing.workspace_size,timing.recipe));
  uint64_t workspace_aligned=(timing.workspace_size+255)&~uint64_t(255);
  if(timing_storage){
    if(workspace_aligned>timing_storage_bytes-512)return -1001;
    timing.workspace=timing_storage;timing.input=timing_storage+workspace_aligned;timing.output=timing.input+256;
  }else{
    if(timing.workspace_size)CK(synDeviceMalloc(device,timing.workspace_size,0,0,&timing.workspace));
    CK(synDeviceMalloc(device,256,0,0,&timing.input));
    CK(synDeviceMalloc(device,256,0,0,&timing.output));
  }
  uint64_t ids[2]{};CK(synTensorRetrieveIds(timing.recipe,names,ids,2));
  for(int i=0;i<2;++i)timing.binding[i].tensorId=ids[i];
  CK(synHostMalloc(device,256,0,&timing.host));
  for(int i=0;i<64;++i)((float*)timing.host)[i]=.125f;
  timing.binding[0].pTensorAddress=timing.input;timing.binding[1].pTensorAddress=timing.output;
  CK(synMemCopyAsync(serial_stream,(uint64_t)timing.host,256,timing.input,HOST_TO_DRAM));
  CK(synStreamSynchronize(serial_stream));
  CK(synEventCreate(&timing.gate_begin,device,EVENT_COLLECT_TIME));
  CK(synEventCreate(&timing.begin,device,EVENT_COLLECT_TIME));
  CK(synEventCreate(&timing.end,device,EVENT_COLLECT_TIME));
  return 0;
#undef CK
}

static int align_timing_gate(){
  if(!option(32))return 0;
  for(const auto& c:commands)if(c.kind==2){
    auto ar=reinterpret_cast<decltype(&hcclAllReduce)>(resolve("hcclAllReduce"));
    return ar((void*)timing.input,(void*)timing.output,1,hcclFloat32,hcclSum,(hcclComm_t)c.object,serial_stream);
  }
  return 0; // Single-card recurrence fixture has no communicator.
}
// `gate_query` must equal synBusy; no device span is valid when the gate has drained.
extern "C" int e1_trial(unsigned n,int gate_loops,uint64_t* results) {
  if(!timing.recipe)return -1;
  auto clock=[](){return std::chrono::steady_clock::now();};
  Guard guard;
  int status=synStreamSynchronize(serial_stream);if(status)return status;
  auto start=clock();
  if(gate_loops){
    // The compiled gate is fixed-length; caller chooses whether to use it.
    status=synEventRecord(timing.gate_begin,serial_stream);if(status)return status;
    status=synLaunch(serial_stream,timing.binding,2,timing.workspace,timing.recipe,0);if(status)return status;
    status=align_timing_gate();if(status)return status;
  }
  status=synEventRecord(timing.begin,serial_stream);if(status)return status;
  uint64_t host=0,launch=0,hccl=0;
  status=enqueue(n,&host,&launch,&hccl);if(status)return status;
  auto submitted=clock();
  status=synEventRecord(timing.end,serial_stream);if(status)return status;
  int query=gate_loops?(int)synEventQuery(timing.begin):-1;
  status=synStreamSynchronize(serial_stream);if(status)return status;
  auto finish=clock();
  uint64_t span=0,gate_ns=0;
  status=synEventElapsedTime(&span,timing.begin,timing.end);if(status)return status;
  if(gate_loops){status=synEventElapsedTime(&gate_ns,timing.gate_begin,timing.begin);if(status)return status;}
  results[0]=span;results[1]=host;results[2]=launch;results[3]=hccl;
  results[4]=std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
  results[5]=query;results[6]=gate_ns;
  results[7]=std::chrono::duration_cast<std::chrono::nanoseconds>(submitted-start).count();
  return 0;
}

#include "e1_boundaries.inc"
#include "e1_profile.inc"

// Bounded multi-request diagnostic, not a production allocator reset. Old plans,
// streams and pinned buffers remain owned until process exit (at most 3 plans).
// This preserves every recorded address while allowing context-ladder checks.
static std::vector<std::vector<Command>> retired_commands;
static std::vector<RolloutState> retired_rollouts;
static std::vector<synStreamHandle> retired_streams;
extern "C" int e1_rearm_bounded_diagnostic(){
  std::lock_guard<std::mutex> lock(mutex_);
  if(active.load()||commands.empty()||rejected||timing.recipe||profile_started||!boundary_commands.empty())return -1;
  if(retired_commands.size()>=2)return -2;
  Guard guard;
  int code=synStreamSynchronize(serial_stream);if(code)return code;
  retired_commands.push_back(std::move(commands));commands.clear();
  retired_rollouts.push_back(rollout);rollout=RolloutState{};
  retired_streams.push_back(serial_stream);serial_stream=nullptr;
  recipe_metadata.clear();recorded_events.clear();
  metadata_queries=0;metadata_tensors=0;metadata_images=0;metadata_empty=0;metadata_omitted=0;partial_outputs=0;
  snapshot_bytes=0;launches=collectives=memcopies=external_events=external_waits=syncs=0;
  return 0;
}

#include "native_step.inc"
