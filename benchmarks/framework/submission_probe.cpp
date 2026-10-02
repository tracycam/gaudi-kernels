// Bounded same-process submission ablation, not an owning/general executor.
// All graph/tensor owners must remain live; no allocation/other graph may run
// in a native window. Physical address locks are NOT acquired by this helper.
#include <synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
using Clock=std::chrono::steady_clock;
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static const char* labels[]={"synLaunch","synLaunchWithExternalEvents","synLaunchWithExternalEventsExt","scal_stream_submit","synStreamWaitEvent","synMemCopyAsync","synMemCopyAsyncMultiple","synStreamSynchronize","synDeviceSynchronize","synEventRecord","synMemsetD16Async","synDeviceMalloc","synDeviceFree","hcclAllReduce","hcclAllGather","scal_inside_launch_api","scal_inside_event_record","scal_inside_other_observed_api","scal_outside_observed_api"};
static constexpr unsigned METRICS=sizeof(labels)/sizeof(labels[0]);
struct Metric{std::atomic<uint64_t> count{0},duration{0},errors{0};};static Metric metrics[METRICS];
static thread_local unsigned api_depth=0,outer_api=METRICS;
struct Scope{unsigned id;bool outer;uint64_t start;Scope(unsigned i,bool scal=false):id(i),outer(scal||api_depth==0),start(ns()){if(!scal){if(outer)outer_api=i;++api_depth;}}void end(int status,bool scal=false){if(outer){metrics[id].count++;metrics[id].duration+=ns()-start;metrics[id].errors+=status!=0;}if(!scal){--api_depth;if(!api_depth)outer_api=METRICS;}}};
template<class T>T real(const char* name){void* p=dlsym(RTLD_NEXT,name);if(!p){for(const char* file:{"/usr/lib/habanalabs/libSynapse.so","/usr/lib/habanalabs/libscal.so","/usr/lib/habanalabs/libhccl.so"}){void* h=dlopen(file,RTLD_NOW|RTLD_NOLOAD);if(h){p=dlsym(h,name);if(p)break;}}}if(!p){std::fprintf(stderr,"submission_probe missing %s\n",name);std::abort();}return reinterpret_cast<T>(p);}
struct Launch{unsigned kind=0;uint32_t flags=0;uint64_t workspace=0;synStreamHandle stream=nullptr;synRecipeHandle recipe=nullptr;std::vector<synLaunchTensorInfo> base;std::vector<synLaunchTensorInfoExt> ext;std::vector<std::string> names;std::vector<bool> null_names;void bind(){for(size_t i=0;i<names.size();++i){const char* p=null_names[i]?nullptr:names[i].c_str();base[i].tensorName=p;ext[i].tensorName=p;}}};
struct ScalCall{uint64_t stream;unsigned producer,alignment,parent;int status;uint64_t duration;};
static std::mutex mu;static std::atomic<bool> recording{false},native_window{false};static std::vector<Launch> captured;static std::vector<ScalCall> scal_calls;
static Launch plan;static bool have_plan=false,validated=false;static unsigned rejected=0;static std::string error;
static synEventHandle begin_event=nullptr,end_event=nullptr;
static std::vector<std::pair<uint64_t,size_t>> outputs;
static void reject(unsigned id,bool outer){if(!outer)return;if(recording.load()||native_window.load()){std::lock_guard<std::mutex> g(mu);rejected|=(1u<<id);}}
template<class T>static void capture(unsigned kind,synStreamHandle s,const T* t,unsigned n,uint64_t w,synRecipeHandle r,unsigned ne,unsigned f,bool outer){
 if(!outer||!recording.load())return;std::lock_guard<std::mutex> g(mu);if(!recording.load())return;
 if(ne)rejected|=1u<<31;Launch c;c.kind=kind;c.flags=f;c.workspace=w;c.stream=s;c.recipe=r;c.names.reserve(n);c.null_names.reserve(n);c.base.resize(n);c.ext.resize(n);
 for(unsigned i=0;i<n;++i){c.names.emplace_back(t[i].tensorName?t[i].tensorName:"");c.null_names.push_back(t[i].tensorName==nullptr);auto& b=c.base[i];auto& e=c.ext[i];b.pTensorAddress=e.pTensorAddress=t[i].pTensorAddress;b.tensorId=e.tensorId=t[i].tensorId;b.tensorType=e.tensorType=t[i].tensorType;std::copy(std::begin(t[i].tensorSize),std::end(t[i].tensorSize),b.tensorSize);std::copy(std::begin(t[i].tensorSize),std::end(t[i].tensorSize),e.tensorSize);if(t[i].tensorType!=DATA_TENSOR)rejected|=1u<<30;}
 captured.push_back(std::move(c));
}
extern "C" synStatus synLaunch(const synStreamHandle s,const synLaunchTensorInfo* t,const uint32_t n,uint64_t w,const synRecipeHandle r,uint32_t f){static auto fn=real<decltype(&synLaunch)>("synLaunch");Scope a(0);capture(0,s,t,n,w,r,0,f,a.outer);auto rc=fn(s,t,n,w,r,f);if(rc!=synSuccess)reject(0,a.outer);a.end(rc);return rc;}
extern "C" synStatus synLaunchWithExternalEvents(const synStreamHandle s,const synLaunchTensorInfo* t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle* e,const uint32_t ne,uint32_t f){static auto fn=real<decltype(&synLaunchWithExternalEvents)>("synLaunchWithExternalEvents");Scope a(1);capture(1,s,t,n,w,r,ne,f,a.outer);auto rc=fn(s,t,n,w,r,e,ne,f);if(rc!=synSuccess)reject(1,a.outer);a.end(rc);return rc;}
extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt* t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle* e,const uint32_t ne,uint32_t f){static auto fn=real<decltype(&synLaunchWithExternalEventsExt)>("synLaunchWithExternalEventsExt");Scope a(2);capture(2,s,t,n,w,r,ne,f,a.outer);auto rc=fn(s,t,n,w,r,e,ne,f);if(rc!=synSuccess)reject(2,a.outer);a.end(rc);return rc;}
extern "C" int scal_stream_submit(void* stream,unsigned producer,unsigned alignment){using Fn=int(*)(void*,unsigned,unsigned);static auto fn=real<Fn>("scal_stream_submit");Scope a(3,true);auto rc=fn(stream,producer,alignment);uint64_t duration=ns()-a.start;unsigned parent=api_depth==0?18:outer_api<=2?15:outer_api==9?16:17;metrics[parent].count++;metrics[parent].duration+=duration;metrics[parent].errors+=rc!=0;if(recording.load()){std::lock_guard<std::mutex> g(mu);scal_calls.push_back({reinterpret_cast<uint64_t>(stream),producer,alignment,parent,rc,duration});}a.end(rc,true);return rc;}
extern "C" synStatus synStreamWaitEvent(const synStreamHandle s,synEventHandle e,const uint32_t f){static auto fn=real<decltype(&synStreamWaitEvent)>("synStreamWaitEvent");Scope a(4);reject(4,a.outer);auto rc=fn(s,e,f);a.end(rc);return rc;}
extern "C" synStatus synMemCopyAsync(const synStreamHandle s,uint64_t src,uint64_t bytes,uint64_t dst,const synDmaDir d){static auto fn=real<decltype(&synMemCopyAsync)>("synMemCopyAsync");Scope a(5);reject(5,a.outer);auto rc=fn(s,src,bytes,dst,d);a.end(rc);return rc;}
extern "C" synStatus synMemCopyAsyncMultiple(const synStreamHandle s,const uint64_t* src,const uint64_t* bytes,const uint64_t* dst,const synDmaDir d,const uint64_t n){static auto fn=real<decltype(&synMemCopyAsyncMultiple)>("synMemCopyAsyncMultiple");Scope a(6);reject(6,a.outer);auto rc=fn(s,src,bytes,dst,d,n);a.end(rc);return rc;}
extern "C" synStatus synStreamSynchronize(const synStreamHandle s){static auto fn=real<decltype(&synStreamSynchronize)>("synStreamSynchronize");Scope a(7);auto rc=fn(s);a.end(rc);return rc;}
extern "C" synStatus synDeviceSynchronize(const synDeviceId d){static auto fn=real<decltype(&synDeviceSynchronize)>("synDeviceSynchronize");Scope a(8);auto rc=fn(d);a.end(rc);return rc;}
extern "C" synStatus synEventRecord(synEventHandle e,const synStreamHandle s){static auto fn=real<decltype(&synEventRecord)>("synEventRecord");Scope a(9);auto rc=fn(e,s);a.end(rc);return rc;}
extern "C" synStatus synMemsetD16Async(uint64_t p,const uint16_t v,const size_t n,const synStreamHandle s){static auto fn=real<decltype(&synMemsetD16Async)>("synMemsetD16Async");Scope a(10);reject(10,a.outer);auto rc=fn(p,v,n,s);a.end(rc);return rc;}
extern "C" synStatus synDeviceMalloc(const synDeviceId d,const uint64_t n,uint64_t req,const uint32_t f,uint64_t* p){static auto fn=real<decltype(&synDeviceMalloc)>("synDeviceMalloc");Scope a(11);reject(11,a.outer);auto rc=fn(d,n,req,f,p);a.end(rc);return rc;}
extern "C" synStatus synDeviceFree(const synDeviceId d,const uint64_t p,const uint32_t f){static auto fn=real<decltype(&synDeviceFree)>("synDeviceFree");Scope a(12);reject(12,a.outer);auto rc=fn(d,p,f);a.end(rc);return rc;}
extern "C" hcclResult_t hcclAllReduce(const void* src,void* dst,size_t n,hcclDataType_t t,hcclRedOp_t op,hcclComm_t c,void* s){static auto fn=real<decltype(&hcclAllReduce)>("hcclAllReduce");Scope a(13);reject(13,a.outer);auto rc=fn(src,dst,n,t,op,c,s);a.end(rc);return rc;}
extern "C" hcclResult_t hcclAllGather(const void* src,void* dst,size_t n,hcclDataType_t t,hcclComm_t c,void* s){static auto fn=real<decltype(&hcclAllGather)>("hcclAllGather");Scope a(14);reject(14,a.outer);auto rc=fn(src,dst,n,t,c,s);a.end(rc);return rc;}
extern "C" unsigned submission_metric_count(){return METRICS;}
extern "C" const char* submission_metric_name(unsigned i){return i<METRICS?labels[i]:"invalid";}
extern "C" void submission_snapshot(uint64_t* out){for(unsigned i=0;i<METRICS;++i){out[3*i]=metrics[i].count.load();out[3*i+1]=metrics[i].duration.load();out[3*i+2]=metrics[i].errors.load();}}
extern "C" const char* submission_error(){return error.c_str();}
extern "C" int submission_init(unsigned device){if(begin_event)return 0;auto create=real<decltype(&synEventCreate)>("synEventCreate");auto rc=create(&begin_event,device,EVENT_COLLECT_TIME);if(rc!=synSuccess)return rc;return create(&end_event,device,EVENT_COLLECT_TIME);}
extern "C" int submission_record_begin(){std::lock_guard<std::mutex> g(mu);if(recording.load()||native_window.load())return -1;captured.clear();scal_calls.clear();rejected=0;error.clear();recording=true;return 0;}
static bool same(const Launch& a,const Launch& b){
 auto fail=[](std::string why){error=std::move(why);return false;};
 if(a.kind!=b.kind)return fail("launch API variant changed");if(a.flags!=b.flags)return fail("launch flags changed");
 if(a.workspace!=b.workspace)return fail("workspace changed "+std::to_string(a.workspace)+" -> "+std::to_string(b.workspace));
 if(a.recipe!=b.recipe)return fail("recipe handle changed");if(a.stream!=b.stream)return fail("stream handle changed");
 if(a.ext.size()!=b.ext.size())return fail("tensor count changed");
 for(size_t i=0;i<a.ext.size();++i){auto& x=a.ext[i];auto& y=b.ext[i];std::string prefix="tensor["+std::to_string(i)+"] "+a.names[i]+" ";
  if(a.names[i]!=b.names[i]||a.null_names[i]!=b.null_names[i])return fail(prefix+"name/nullness changed");
  if(x.pTensorAddress!=y.pTensorAddress)return fail(prefix+"physical address changed "+std::to_string(x.pTensorAddress)+" -> "+std::to_string(y.pTensorAddress));
  if(x.tensorId!=y.tensorId)return fail(prefix+"tensorId changed");if(x.tensorType!=y.tensorType)return fail(prefix+"tensorType changed");
  for(unsigned d=0;d<HABANA_DIM_MAX;++d)if(x.tensorSize[d]!=y.tensorSize[d])return fail(prefix+"tensorSize["+std::to_string(d)+"] changed "+std::to_string(x.tensorSize[d])+" -> "+std::to_string(y.tensorSize[d]));
 }return true;
}
extern "C" int submission_record_end(unsigned compare){std::lock_guard<std::mutex> g(mu);recording=false;if(rejected||captured.size()!=1){error="unsupported recorded operations mask="+std::to_string(rejected)+", compute recipe calls="+std::to_string(captured.size());validated=false;return -2;}if(compare){if(!have_plan||!same(plan,captured[0])){if(error.empty())error="no previous plan";validated=false;return -3;}validated=true;return 0;}plan=std::move(captured[0]);plan.bind();have_plan=true;validated=false;outputs.clear();return 0;}
extern "C" int submission_dump(const char* file){std::lock_guard<std::mutex> g(mu);FILE* f=std::fopen(file,"w");if(!f)return -1;
 auto dump=[&](const char* label,const Launch& c){std::fprintf(f,"%s validated=%d kind=%u recipe=%p stream=%p workspace=%llu flags=%u tensor_count=%zu rejected_mask=%u\n",label,validated,c.kind,(void*)c.recipe,(void*)c.stream,(unsigned long long)c.workspace,c.flags,c.ext.size(),rejected);for(size_t i=0;i<c.ext.size();++i){auto& t=c.ext[i];std::fprintf(f,"%s tensor[%zu] name=%s null_name=%d address=%llu id=%llu type=%u shape=",label,i,c.names[i].c_str(),bool(c.null_names[i]),(unsigned long long)t.pTensorAddress,(unsigned long long)t.tensorId,unsigned(t.tensorType));for(auto size:t.tensorSize)std::fprintf(f,"%llu,",(unsigned long long)size);std::fprintf(f,"\n");}};
 dump("stored",plan);for(auto& c:captured)if(!c.ext.empty())dump("observed",c);
 std::fprintf(f,"rejection=%s\n",error.c_str());for(auto& c:scal_calls)std::fprintf(f,"scal stream=%llu producer=%u alignment=%u parent=%s status=%d host_ns=%llu\n",(unsigned long long)c.stream,c.producer,c.alignment,labels[c.parent],c.status,(unsigned long long)c.duration);std::fclose(f);return 0;
}
// Select real compiled BF16 outputs via recipe inventory, not unchecked by-name
// metadata. Poisoning happens outside timing to reject a stale-output false pass.
extern "C" int submission_find_outputs(){if(!validated)return -1;uint32_t n=0;auto amount=real<decltype(&synTensorRetrieveLaunchAmount)>("synTensorRetrieveLaunchAmount");auto idsfn=real<decltype(&synTensorRetrieveLaunchIds)>("synTensorRetrieveLaunchIds");auto infofn=real<decltype(&synTensorRetrieveLaunchInfoByIdExt)>("synTensorRetrieveLaunchInfoByIdExt");auto rc=amount(plan.recipe,&n);if(rc!=synSuccess||!n)return -2;std::vector<uint64_t> ids(n);if(idsfn(plan.recipe,ids.data(),n)!=synSuccess)return -3;std::vector<synRetrievedLaunchTensorInfoExt> infos(n);for(unsigned i=0;i<n;++i)infos[i].tensorId=ids[i];if(infofn(plan.recipe,n,infos.data())!=synSuccess)return -4;outputs.clear();for(auto& info:infos){if(info.isInput)continue;if(info.tensorDataType!=syn_type_bf16||info.tensorType!=DATA_TENSOR)continue;bool found=false;for(size_t i=0;i<plan.ext.size();++i){auto& t=plan.ext[i];if((plan.flags&SYN_FLAGS_TENSOR_NAME)?plan.names[i]==info.tensorName:t.tensorId==info.tensorId){size_t elements=1;for(unsigned d=0;d<info.tensorDims;++d){if(!t.tensorSize[d]||elements>8388608/t.tensorSize[d])return -6;elements*=t.tensorSize[d];}outputs.push_back({t.pTensorAddress,elements});found=true;break;}}if(!found)return -7;}return outputs.empty()?-8:0;}
extern "C" int submission_poison_outputs(){if(!validated||outputs.empty()||native_window.load())return -1;for(auto& o:outputs){auto rc=synMemsetD16Async(o.first,0x7fc1,o.second,plan.stream);if(rc!=synSuccess)return rc;}return synStreamSynchronize(plan.stream);}
static synStatus invoke(){switch(plan.kind){case 0:return synLaunch(plan.stream,plan.base.data(),plan.base.size(),plan.workspace,plan.recipe,plan.flags);case 1:return synLaunchWithExternalEvents(plan.stream,plan.base.data(),plan.base.size(),plan.workspace,plan.recipe,nullptr,0,plan.flags);default:return synLaunchWithExternalEventsExt(plan.stream,plan.ext.data(),plan.ext.size(),plan.workspace,plan.recipe,nullptr,0,plan.flags);}}
extern "C" int submission_native(unsigned repeats,uint64_t* timing,uint64_t* loop_metrics){if(!validated||!begin_event||!repeats||repeats>1000||recording.load()||native_window.exchange(true))return -1;rejected=0;uint64_t wall=ns();synStatus rc=synEventRecord(begin_event,plan.stream);uint64_t before[METRICS*3];submission_snapshot(before);uint64_t host=ns();for(unsigned i=0;i<repeats&&rc==synSuccess;++i)rc=invoke();timing[0]=ns()-host;submission_snapshot(loop_metrics);for(unsigned i=0;i<METRICS*3;++i)loop_metrics[i]-=before[i];if(rc==synSuccess)rc=synEventRecord(end_event,plan.stream);if(rc==synSuccess)rc=real<decltype(&synEventSynchronize)>("synEventSynchronize")(end_event);timing[1]=ns()-wall;if(rc==synSuccess)rc=real<decltype(&synEventElapsedTime)>("synEventElapsedTime")(&timing[2],begin_event,end_event);native_window=false;timing[3]=rejected;return rc==synSuccess&&rejected?-9:rc;}
