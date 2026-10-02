// SPDX-License-Identifier: Apache-2.0
// Research-only submission scope. Never replays or edits vendor command bytes.
// Mode0 records calls unchanged. Mode1 keeps the latest producer index per
// SCAL stream and flushes at scope end/before blocking waits, with bounded spans.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
using Submit=int(*)(void*,unsigned,unsigned);
using Wait=int(*)(void*,uint64_t,uint64_t);
static long tid(){return syscall(SYS_gettid);}
template<class T>T resolve(const char*name){void*p=dlsym(RTLD_NEXT,name);if(!p){void*h=dlopen("/usr/lib/habanalabs/libscal.so",RTLD_NOW|RTLD_NOLOAD);if(h)p=dlsym(h,name);}if(!p){std::fprintf(stderr,"missing %s\n",name);std::abort();}return reinterpret_cast<T>(p);}
struct Pending{void*stream;unsigned first,pi,alignment;};
struct State{bool active=false;int mode=0;FILE*file=nullptr;std::vector<Pending>pending;uint64_t requested=0,issued=0,flushes=0,wait_flushes=0;};
static thread_local State state;
static std::atomic<long>owner{0};static std::atomic<uint64_t>foreign{0};
static int flush(const char*reason){
 auto&s=state;if(s.pending.empty())return 0;++s.flushes;
 static auto real=resolve<Submit>("scal_stream_submit");
 for(auto&p:s.pending){int rc=real(p.stream,p.pi,p.alignment);++s.issued;
  if(s.file)std::fprintf(s.file,"{\"event\":\"flush\",\"reason\":\"%s\",\"stream\":%llu,\"first_pi\":%u,\"last_pi\":%u,\"alignment\":%u,\"status\":%d}\n",reason,(unsigned long long)p.stream,p.first,p.pi,p.alignment,rc);
  if(rc){s.pending.clear();return rc;}
 }s.pending.clear();return 0;
}
extern "C" int ng_scope_begin(int mode,const char*path){
 if(state.active||owner.load()||mode<0||mode>2)return -1;
 if(path&&*path){state.file=std::fopen(path,"w");if(!state.file)return -2;}
 state.active=true;state.mode=mode;state.requested=state.issued=state.flushes=state.wait_flushes=0;state.pending.clear();foreign=0;owner=tid();return 0;
}
extern "C" int ng_scope_end(uint64_t*out){
 if(!state.active||owner!=tid())return -1;
 int rc=flush("scope_end");
 out[0]=state.requested;out[1]=state.issued;out[2]=state.flushes;out[3]=state.wait_flushes;out[4]=foreign.load();
 if(state.file){std::fclose(state.file);state.file=nullptr;}state.active=false;owner=0;return rc;
}
extern "C" int scal_stream_submit(void*stream,unsigned pi,unsigned alignment){
 static auto real=resolve<Submit>("scal_stream_submit");auto&s=state;
 if(!s.active){if(owner.load()&&owner.load()!=tid())++foreign;return real(stream,pi,alignment);}
 ++s.requested;
 if(s.file)std::fprintf(s.file,"{\"event\":\"request\",\"stream\":%llu,\"pi\":%u,\"alignment\":%u,\"tid\":%ld}\n",(unsigned long long)stream,pi,alignment,tid());
 if(!s.mode){++s.issued;return real(stream,pi,alignment);}
 for(auto&p:s.pending)if(p.stream==stream){
  if(pi<p.pi||alignment!=p.alignment||uint64_t(pi)-p.first>4096){int rc=flush("bounded_span");if(rc)return rc;break;}
  p.pi=pi;return 0;
 }
 if(s.pending.size()>=32){int rc=flush("bounded_streams");if(rc)return rc;}
 s.pending.push_back({stream,pi,pi,alignment});return 0;
}
static int waiting(const char*name,void*handle,uint64_t target,uint64_t timeout){
 auto&s=state;auto real=resolve<Wait>(name);int rc;
 if(s.file)std::fprintf(s.file,"{\"event\":\"wait\",\"api\":\"%s\",\"target\":%llu,\"timeout\":%llu}\n",name,(unsigned long long)target,(unsigned long long)timeout);
 if(s.active&&s.mode==2&&timeout==0){
  rc=real(handle,target,timeout);
  if(rc==0||s.pending.empty())return rc;
  // A not-ready query can require progress of our queued work. Publish it
  // before the caller takes its ordinary blocking/resource-allocation path.
  ++s.wait_flushes;int flushed=flush(name);if(flushed)return flushed;
  return real(handle,target,timeout);
 }
 if(s.active&&s.mode&&!s.pending.empty()){++s.wait_flushes;int flushed=flush(name);if(flushed)return flushed;}
 rc=real(handle,target,timeout);
 if(s.file)std::fprintf(s.file,"{\"event\":\"wait_return\",\"status\":%d}\n",rc);
 return rc;
}
extern "C" int scal_completion_group_wait(void*h,uint64_t target,uint64_t timeout){return waiting("scal_completion_group_wait",h,target,timeout);}
extern "C" int scal_completion_group_wait_always_interupt(void*h,uint64_t target,uint64_t timeout){return waiting("scal_completion_group_wait_always_interupt",h,target,timeout);}
extern "C" int scal_host_fence_counter_wait(void*h,uint64_t credits,uint64_t timeout){return waiting("scal_host_fence_counter_wait",h,credits,timeout);}
