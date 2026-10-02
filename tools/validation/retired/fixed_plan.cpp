// Experimental replay of a validated, fixed-address Synapse/HCCL command list.
// This still calls the CPU APIs for each command; it is NOT one device submission.
// Caller synchronizes bridge work before capture/replay; replay synchronizes at end.
// Reject changing descriptors, memcpy, external tensor events, and missing collectives.
#include <synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static void* resolve(const char* name){
 void* p=dlsym(RTLD_NEXT,name);
 if(!p){void* h=dlopen("/usr/lib/habanalabs/libSynapse.so",RTLD_NOW|RTLD_NOLOAD);if(h)p=dlsym(h,name);}
 if(!p){fprintf(stderr,"FIXED_PLAN missing %s\n",name);abort();}return p;
}
#define REAL(f) static auto real=reinterpret_cast<decltype(&f)>(resolve(#f))
struct Command{
 int kind=0,type=0,op=0;uint32_t flags=0;
 void* stream=nullptr;void* object=nullptr;void* src=nullptr;void* dst=nullptr;
 uint64_t count=0,workspace=0;
 std::vector<synLaunchTensorInfoExt> tensors;
 std::vector<std::string> names;
};
struct Plan{
 std::vector<Command> commands;
 std::map<void*,synEventHandle> events;
 bool validated=false;unsigned launches=0,collectives=0,external_waits=0;
 synStreamHandle serial_stream=nullptr;
};
static std::map<uint64_t,Plan> plans;
static std::vector<Command> recording;
static std::mutex mutex_;
static std::atomic<bool> active{false};
static uint64_t key_=0;
static bool rejected=false;
static std::string mismatch;
extern "C" const char* fixed_plan_error(){return mismatch.c_str();}
static thread_local unsigned depth=0;
struct Guard{Guard(){++depth;}~Guard(){--depth;}};
static void add(Command c){if(depth||!active.load())return;std::lock_guard<std::mutex> l(mutex_);if(active.load())recording.push_back(std::move(c));}
static void reject(){if(!depth&&active.load()){std::lock_guard<std::mutex> l(mutex_);rejected=true;}}

extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt* ts,
 const uint32_t n,uint64_t workspace,const synRecipeHandle recipe,synEventHandle* events,const uint32_t ne,uint32_t flags){
 REAL(synLaunchWithExternalEventsExt);
 if(!depth&&active.load()){
  if(ne)reject();
  Command c;c.kind=1;c.stream=(void*)s;c.object=(void*)recipe;c.workspace=workspace;c.flags=flags;
  c.tensors.assign(ts,ts+n);for(unsigned i=0;i<n;++i)c.names.emplace_back(ts[i].tensorName?ts[i].tensorName:"");add(std::move(c));
 }
 Guard g;return real(s,ts,n,workspace,recipe,events,ne,flags);
}
extern "C" hcclResult_t hcclAllReduce(const void* src,void* dst,size_t count,hcclDataType_t type,hcclRedOp_t op,hcclComm_t comm,void* stream){
 REAL(hcclAllReduce);Command c;c.kind=2;c.src=(void*)src;c.dst=dst;c.count=count;c.type=type;c.op=op;c.object=(void*)comm;c.stream=stream;add(std::move(c));Guard g;return real(src,dst,count,type,op,comm,stream);
}
extern "C" hcclResult_t hcclAllGather(const void* src,void* dst,size_t count,hcclDataType_t type,hcclComm_t comm,void* stream){
 REAL(hcclAllGather);Command c;c.kind=3;c.src=(void*)src;c.dst=dst;c.count=count;c.type=type;c.object=(void*)comm;c.stream=stream;add(std::move(c));Guard g;return real(src,dst,count,type,comm,stream);
}
extern "C" synStatus synEventRecord(synEventHandle event,const synStreamHandle stream){
 REAL(synEventRecord);Command c;c.kind=4;c.object=(void*)event;c.stream=(void*)stream;add(std::move(c));Guard g;return real(event,stream);
}
extern "C" synStatus synStreamWaitEvent(const synStreamHandle stream,synEventHandle event,const uint32_t flags){
 REAL(synStreamWaitEvent);Command c;c.kind=5;c.object=(void*)event;c.stream=(void*)stream;c.flags=flags;add(std::move(c));Guard g;return real(stream,event,flags);
}
extern "C" synStatus synMemCopyAsync(const synStreamHandle stream,uint64_t src,uint64_t n,uint64_t dst,const synDmaDir direction){
 REAL(synMemCopyAsync);reject();Guard g;return real(stream,src,n,dst,direction);
}
extern "C" synStatus synMemCopyAsyncMultiple(const synStreamHandle stream,const uint64_t* src,const uint64_t* n,const uint64_t* dst,const synDmaDir direction,const uint64_t count){
 REAL(synMemCopyAsyncMultiple);reject();Guard g;return real(stream,src,n,dst,direction,count);
}
static bool equal(const std::vector<Command>& a,const std::vector<Command>& b){
 auto fail=[](std::string text){mismatch=std::move(text);return false;};
 if(a.size()!=b.size())return fail("command count "+std::to_string(a.size())+" vs "+std::to_string(b.size()));
 std::map<void*,unsigned> ea,eb;
 for(size_t i=0;i<a.size();++i){
  const auto& x=a[i];const auto& y=b[i];
  if(x.kind!=y.kind||x.stream!=y.stream||x.type!=y.type||x.op!=y.op||x.flags!=y.flags||x.src!=y.src||x.dst!=y.dst||x.count!=y.count||x.workspace!=y.workspace)return fail("command "+std::to_string(i)+" kind="+std::to_string(x.kind)+" changed fields stream="+std::to_string(x.stream!=y.stream)+" src="+std::to_string(x.src!=y.src)+" dst="+std::to_string(x.dst!=y.dst)+" workspace="+std::to_string(x.workspace!=y.workspace));
  if(x.kind==4||x.kind==5){
   if(!ea.count(x.object))ea[x.object]=ea.size()+1;
   if(!eb.count(y.object))eb[y.object]=eb.size()+1;
   if(ea[x.object]!=eb[y.object])return false;
  }else if(x.object!=y.object)return fail("command "+std::to_string(i)+" recipe/comm changed");
  if(x.names!=y.names||x.tensors.size()!=y.tensors.size())return fail("command "+std::to_string(i)+" tensor names/count changed");
  for(size_t j=0;j<x.tensors.size();++j){
   auto p=x.tensors[j],q=y.tensors[j];
   if(p.pTensorAddress!=q.pTensorAddress||p.tensorId!=q.tensorId||p.tensorType!=q.tensorType||memcmp(p.tensorSize,q.tensorSize,sizeof(p.tensorSize)))return fail("command "+std::to_string(i)+" tensor "+std::to_string(j)+" "+x.names[j]+" changed address="+std::to_string(p.pTensorAddress!=q.pTensorAddress)+" id="+std::to_string(p.tensorId!=q.tensorId)+" shape="+std::to_string(memcmp(p.tensorSize,q.tensorSize,sizeof(p.tensorSize))!=0));
  }
 }
 return true;
}
extern "C" int fixed_plan_begin(uint64_t key){
 std::lock_guard<std::mutex> l(mutex_);if(active.load())return -1;
 key_=key;recording.clear();rejected=false;active.store(true);return 0;
}
extern "C" int fixed_plan_end(){
 std::lock_guard<std::mutex> l(mutex_);active.store(false);
 if(rejected||recording.empty())return -2;
 auto it=plans.find(key_);
 if(it!=plans.end()){
  bool match=false;
  if(it->second.serial_stream){
   std::vector<Command> a,b;
   for(const auto&c:it->second.commands)if(c.kind<=3)a.push_back(c);
   for(const auto&c:recording)if(c.kind<=3)b.push_back(c);
   match=equal(a,b);
  }else match=equal(it->second.commands,recording);
  if(!match)return -3;
  it->second.validated=true;return 1;
 }
 Plan p;p.commands=std::move(recording);
 if(getenv("FIXED_PLAN_SERIAL")&&std::strcmp(getenv("FIXED_PLAN_SERIAL"),"1")==0){
  auto create_stream=reinterpret_cast<decltype(&synStreamCreateGeneric)>(resolve("synStreamCreateGeneric"));
  if(create_stream(&p.serial_stream,0,0)!=synSuccess)return -6;
 }
 std::set<void*> seen;
 auto create=reinterpret_cast<decltype(&synEventCreate)>(resolve("synEventCreate"));
 for(auto& c:p.commands){
  if(c.kind==1){++p.launches;for(size_t i=0;i<c.tensors.size();++i)c.tensors[i].tensorName=c.names[i].c_str();}
  if(c.kind==2||c.kind==3)++p.collectives;
  if(c.kind==4){seen.insert(c.object);if(!p.events.count(c.object)){
   synEventHandle e=nullptr;auto status=create(&e,0,0);if(status!=synSuccess)return -4;p.events[c.object]=e;
  }}
  if(c.kind==5&&!seen.count(c.object))++p.external_waits;
 }
 if(!p.launches||!p.collectives)return -5;
 plans.emplace(key_,std::move(p));return 0;
}
extern "C" int fixed_plan_stats(uint64_t key,uint64_t* out){
 auto it=plans.find(key);if(it==plans.end())return -1;
 auto&p=it->second;out[0]=p.commands.size();out[1]=p.launches;out[2]=p.collectives;out[3]=p.external_waits;out[4]=p.validated;return 0;
}
extern "C" int fixed_plan_replay(uint64_t key){
 auto it=plans.find(key);if(it==plans.end()||!it->second.validated)return -1;
 auto& p=it->second;
 auto launch=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(resolve("synLaunchWithExternalEventsExt"));
 auto ar=reinterpret_cast<decltype(&hcclAllReduce)>(resolve("hcclAllReduce"));
 auto ag=reinterpret_cast<decltype(&hcclAllGather)>(resolve("hcclAllGather"));
 auto rec=reinterpret_cast<decltype(&synEventRecord)>(resolve("synEventRecord"));
 auto wait=reinterpret_cast<decltype(&synStreamWaitEvent)>(resolve("synStreamWaitEvent"));
 auto sync=reinterpret_cast<decltype(&synStreamSynchronize)>(resolve("synStreamSynchronize"));
 std::set<void*> seen,streams;Guard guard;
 for(auto& c:p.commands){
  int status=0;auto stream=p.serial_stream?p.serial_stream:(synStreamHandle)c.stream;
  streams.insert((void*)stream);
  switch(c.kind){
   case 1:status=launch(stream,c.tensors.data(),c.tensors.size(),c.workspace,(synRecipeHandle)c.object,nullptr,0,c.flags);break;
   case 2:status=ar(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclRedOp_t)c.op,(hcclComm_t)c.object,(void*)stream);break;
   case 3:status=ag(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclComm_t)c.object,(void*)stream);break;
   case 4:if(!p.serial_stream)status=rec(p.events.at(c.object),stream);seen.insert(c.object);break;
   case 5:if(!p.serial_stream&&seen.count(c.object))status=wait(stream,p.events.at(c.object),c.flags);break;
  }
  if(status)return status;
 }
 for(auto s:streams){int status=sync((synStreamHandle)s);if(status)return status;}
 return 0;
}
