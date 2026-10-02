// SPDX-License-Identifier: Apache-2.0
// Interposes public SDK calls only. Reference calls still execute normally;
// a rejected recording can never become an instantiated replay graph.
#include "graph.h"
#include "capture_internal.h"
#include <synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace {
std::mutex mutex;gkg_graph*active=nullptr;void*stream=nullptr;std::thread::id owner;bool have_owner=false;
std::atomic<bool>recording{false};
gkg_status rejected=GKG_OK;std::string reason;std::map<std::string,uint64_t>counts;std::map<void*,std::string>recipes;uint64_t sequence=0;
thread_local unsigned depth=0;struct Guard{Guard(){++depth;}~Guard(){--depth;}};
template<class T>T resolve(const char*n){void*p=dlsym(RTLD_NEXT,n);if(!p)for(const char*file:{"/usr/lib/habanalabs/libSynapse.so","/usr/lib/habanalabs/libhcl.so"}){void*h=dlopen(file,RTLD_NOW|RTLD_NOLOAD);if(h)p=dlsym(h,n);if(p)break;}if(!p){std::fprintf(stderr,"native capture missing %s\n",n);std::abort();}return reinterpret_cast<T>(p);}
#define REAL(F) static auto real=resolve<decltype(&F)>(#F)
void fail(gkg_status s,const std::string&m){if(!rejected){rejected=s;reason=m;}}
bool enter(const char*name,void*submitted=nullptr){
 if(!active||depth)return false;
 ++counts[name];
 auto thread=std::this_thread::get_id();if(!have_owner){owner=thread;have_owner=true;}
 if(owner!=thread){fail(GKG_UNSUPPORTED,"multiple host submission threads");return false;}
 if(submitted&&submitted!=stream){fail(GKG_UNSUPPORTED,"multiple capture streams");return false;}
 return !rejected;
}
void deny(const char*name){if(depth||!recording)return;std::lock_guard<std::mutex>lock(mutex);if(enter(name))fail(GKG_UNSUPPORTED,std::string("forbidden during capture: ")+name);}
void checked(gkg_status s){if(s)fail(s,gkg_error(active));}
unsigned bits(synDataType d){switch(d){case syn_type_bf16:case syn_type_fp16:case syn_type_int16:case syn_type_uint16:case syn_type_ufp16:return 16;case syn_type_single:case syn_type_int32:case syn_type_uint32:case syn_type_tf32:case syn_type_hb_float:return 32;case syn_type_int64:case syn_type_uint64:return 64;case syn_type_fixed:case syn_type_uint8:case syn_type_fp8_143:case syn_type_fp8_152:return 8;case syn_type_packed_mxfp4:case syn_type_packed_nf4:case syn_type_int4:case syn_type_uint4:return 4;default:return 0;}}
void launch(const char*api,void*s,synRecipeHandle recipe,const synLaunchTensorInfoExt*t,uint32_t count,uint32_t flags,uint32_t external_events){
 if(depth||!recording)return;
 std::lock_guard<std::mutex>lock(mutex);if(!enter(api,s))return;Guard guard;
 if(external_events||flags&~SYN_FLAGS_TENSOR_NAME){fail(GKG_UNSUPPORTED,"external launch events/unknown flags");return;}
 uint32_t amount=0;if(synTensorRetrieveLaunchAmount(recipe,&amount)!=synSuccess||!amount||amount>65536){fail(GKG_RUNTIME,"capture recipe inventory");return;}
 std::vector<uint64_t>ids(amount);std::vector<synRetrievedLaunchTensorInfoExt>info(amount);
 if(synTensorRetrieveLaunchIds(recipe,ids.data(),amount)!=synSuccess){fail(GKG_RUNTIME,"capture launch ids");return;}
 for(uint32_t i=0;i<amount;++i)info[i].tensorId=ids[i];
 if(synTensorRetrieveLaunchInfoByIdExt(recipe,amount,info.data())!=synSuccess){fail(GKG_RUNTIME,"capture launch metadata");return;}
 std::vector<std::string>buffers(amount);std::vector<uint64_t>offsets(amount);std::vector<gkg_binding>bindings(amount);
 // All reads first, so outputs of the same node cannot legitimize a missing
 // initial input. Compiled-away submitted bindings are intentionally omitted.
 for(int input:{1,0})for(uint32_t i=0;i<amount;++i){auto&m=info[i];if(bool(m.isInput)!=bool(input))continue;
  if(m.tensorType!=DATA_TENSOR||!m.tensorDims||m.tensorDims>HABANA_DIM_MAX){fail(GKG_UNSUPPORTED,"capture requires static data tensors");return;}
  const synLaunchTensorInfoExt*found=nullptr;for(uint32_t j=0;j<count;++j){bool match=flags&SYN_FLAGS_TENSOR_NAME?(t[j].tensorName&&std::strcmp(t[j].tensorName,m.tensorName)==0):t[j].tensorId==m.tensorId;if(match){if(found){fail(GKG_INVALID,"duplicate live launch binding");return;}found=&t[j];}}
  if(!found){fail(GKG_INVALID,"missing compiled launch binding");return;}uint64_t size=bits(m.tensorDataType);if(!size){fail(GKG_UNSUPPORTED,"unsupported captured tensor dtype");return;}
  for(uint32_t d=0;d<m.tensorDims;++d){if(m.tensorMaxSize[d]!=m.tensorMinSize[d]||!m.tensorMaxSize[d]||size>UINT64_MAX/m.tensorMaxSize[d]){fail(GKG_UNSUPPORTED,"dynamic/overflowing capture shape");return;}size*=m.tensorMaxSize[d];}
  if(size>UINT64_MAX-7){fail(GKG_RANGE,"capture size overflow");return;}size=(size+7)/8;char name[1024];uint64_t offset=0;checked(gkg_capture_address(active,found->pTensorAddress,size,!input,name,sizeof(name),&offset));if(rejected)return;buffers[i]=name;offsets[i]=offset;
 }
 auto r=recipes.find(recipe);if(r==recipes.end()){auto name="captured-recipe-"+std::to_string(++sequence);checked(gkg_recipe(active,name.c_str(),recipe));if(rejected)return;r=recipes.emplace(recipe,name).first;}
 for(uint32_t i=0;i<amount;++i)bindings[i]={sizeof(gkg_binding),1,info[i].tensorName,buffers[i].c_str(),offsets[i]};
 checked(gkg_launch(active,r->second.c_str(),bindings.data(),amount));
}
void collective(const char*api,gkg_collective kind,const void*src,void*dst,size_t count,hcclDataType_t type,hcclRedOp_t op,hcclComm_t comm,void*s){
 if(depth||!recording)return;
 std::lock_guard<std::mutex>lock(mutex);if(!enter(api,s))return;Guard guard;
 gkg_dtype dtype;unsigned width;if(type==hcclFloat){dtype=GKG_F32;width=4;}else if(type==hcclBfloat16){dtype=GKG_BF16;width=2;}else if(type==hcclInt32){dtype=GKG_I32;width=4;}else if(type==hcclUint8){dtype=GKG_U8;width=1;}else{fail(GKG_UNSUPPORTED,"captured HCCL dtype");return;}
 if(op!=hcclSum){fail(GKG_UNSUPPORTED,"captured reduction op");return;}int ranks=0;if(hcclCommCount(comm,&ranks)!=hcclSuccess||ranks<1||!count||count>UINT64_MAX/width/uint64_t(ranks)){fail(GKG_RANGE,"captured collective size");return;}
 char cname[1024],source[1024],dest[1024];uint64_t so=0,dso=0;checked(gkg_capture_communicator(active,comm,cname,sizeof(cname)));if(rejected)return;
 uint64_t bytes=count*width;checked(gkg_capture_address(active,(uint64_t)src,kind==GKG_REDUCE_SCATTER?bytes*ranks:bytes,0,source,sizeof(source),&so));if(rejected)return;
 checked(gkg_capture_address(active,(uint64_t)dst,kind==GKG_ALL_GATHER?bytes*ranks:bytes,1,dest,sizeof(dest),&dso));if(rejected)return;
 checked(gkg_collect(active,kind,cname,source,so,dest,dso,count,dtype));
}
}
extern "C" gkg_status gkg_capture_begin(gkg_graph*g,void*s){std::lock_guard<std::mutex>lock(mutex);if(!g||!s||active)return GKG_STATE;active=g;stream=s;have_owner=false;rejected=GKG_OK;reason.clear();counts.clear();recipes.clear();recording=true;return GKG_OK;}
extern "C" gkg_status gkg_capture_end(gkg_graph*g,const char*file){std::lock_guard<std::mutex>lock(mutex);if(g!=active||!g)return GKG_STATE;
 if(counts.empty())fail(GKG_INVALID,"empty recording");
 auto status=rejected;recording=false;active=nullptr;stream=nullptr;
 if(file){std::ofstream f(file);if(!f){status=GKG_INVALID;reason="cannot write capture report";}else{f<<"{\"status\":"<<status<<",\"calls\":{";bool first=true;for(auto&kv:counts){if(!first)f<<',';first=false;f<<'\"'<<kv.first<<"\":"<<kv.second;}f<<"},\"strict\":true,\"host_io_allowed\":false,\"sync_allowed\":false,\"scope\":\"one public-API host submission thread and stream\"}\n";}}
 if(status)gkg_capture_fail(g,status,reason.c_str());
 return status;
}
extern "C" synStatus synLaunch(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,uint32_t flags){REAL(synLaunch);std::vector<synLaunchTensorInfoExt>ext;
 if(!depth&&recording){ext.resize(n);for(uint32_t i=0;i<n;++i){ext[i].tensorName=t[i].tensorName;ext[i].tensorId=t[i].tensorId;ext[i].pTensorAddress=t[i].pTensorAddress;ext[i].tensorType=t[i].tensorType;std::copy(t[i].tensorSize,t[i].tensorSize+HABANA_DIM_MAX,ext[i].tensorSize);}launch("synLaunch",s,r,ext.data(),n,flags,0);}Guard guard;return real(s,t,n,w,r,flags);
}
extern "C" synStatus synLaunchWithExternalEventsExt(const synStreamHandle s,const synLaunchTensorInfoExt*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*events,const uint32_t ne,uint32_t flags){REAL(synLaunchWithExternalEventsExt);launch("synLaunchWithExternalEventsExt",s,r,t,n,flags,ne);Guard guard;return real(s,t,n,w,r,events,ne,flags);}
extern "C" synStatus synLaunchWithExternalEvents(const synStreamHandle s,const synLaunchTensorInfo*t,const uint32_t n,uint64_t w,const synRecipeHandle r,synEventHandle*events,const uint32_t ne,uint32_t flags){REAL(synLaunchWithExternalEvents);std::vector<synLaunchTensorInfoExt>ext;
 if(!depth&&recording){ext.resize(n);for(uint32_t i=0;i<n;++i){ext[i].tensorName=t[i].tensorName;ext[i].tensorId=t[i].tensorId;ext[i].pTensorAddress=t[i].pTensorAddress;ext[i].tensorType=t[i].tensorType;std::copy(t[i].tensorSize,t[i].tensorSize+HABANA_DIM_MAX,ext[i].tensorSize);}launch("synLaunchWithExternalEvents",s,r,ext.data(),n,flags,ne);}Guard guard;return real(s,t,n,w,r,events,ne,flags);}
extern "C" synStatus synRecipeDestroy(synRecipeHandle r){REAL(synRecipeDestroy);if(!depth&&recording){std::lock_guard<std::mutex>lock(mutex);recipes.erase(r);}Guard guard;return real(r);}
extern "C" hcclResult_t hcclAllReduce(const void*a,void*b,size_t n,hcclDataType_t d,hcclRedOp_t op,hcclComm_t c,void*s){REAL(hcclAllReduce);collective("hcclAllReduce",GKG_ALL_REDUCE,a,b,n,d,op,c,s);Guard g;return real(a,b,n,d,op,c,s);}
extern "C" hcclResult_t hcclAllGather(const void*a,void*b,size_t n,hcclDataType_t d,hcclComm_t c,void*s){REAL(hcclAllGather);collective("hcclAllGather",GKG_ALL_GATHER,a,b,n,d,hcclSum,c,s);Guard g;return real(a,b,n,d,c,s);}
extern "C" hcclResult_t hcclReduceScatter(const void*a,void*b,size_t n,hcclDataType_t d,hcclRedOp_t op,hcclComm_t c,void*s){REAL(hcclReduceScatter);collective("hcclReduceScatter",GKG_REDUCE_SCATTER,a,b,n,d,op,c,s);Guard g;return real(a,b,n,d,op,c,s);}
extern "C" synStatus synMemCopyAsync(const synStreamHandle s,uint64_t a,uint64_t bytes,uint64_t b,const synDmaDir direction){REAL(synMemCopyAsync);
 if(!depth&&recording){std::lock_guard<std::mutex>lock(mutex);if(enter("synMemCopyAsync",s)){if(direction!=DRAM_TO_DRAM)fail(GKG_UNSUPPORTED,"host I/O must be declared outside capture");else{Guard guard;char src[1024],dst[1024];uint64_t so,dso;checked(gkg_capture_address(active,a,bytes,0,src,sizeof(src),&so));if(!rejected)checked(gkg_capture_address(active,b,bytes,1,dst,sizeof(dst),&dso));if(!rejected)checked(gkg_copy(active,src,so,dst,dso,bytes));}}}Guard guard;return real(s,a,bytes,b,direction);
}
#define SYN_DENY(F,PARAMS,ARGS) extern "C" synStatus F PARAMS {REAL(F);deny(#F);Guard guard;return real ARGS;}
SYN_DENY(synStreamSynchronize,(const synStreamHandle s),(s))
SYN_DENY(synEventSynchronize,(const synEventHandle e),(e))
SYN_DENY(synDeviceSynchronize,(const synDeviceId d),(d))
SYN_DENY(synEventRecord,(synEventHandle e,const synStreamHandle s),(e,s))
SYN_DENY(synStreamWaitEvent,(const synStreamHandle s,synEventHandle e,const uint32_t flags),(s,e,flags))
SYN_DENY(synMemCopyAsyncMultiple,(const synStreamHandle s,const uint64_t*a,const uint64_t*n,const uint64_t*b,const synDmaDir d,const uint64_t count),(s,a,n,b,d,count))
SYN_DENY(synMemsetD32Async,(uint64_t a,const uint32_t value,const size_t n,const synStreamHandle s),(a,value,n,s))
SYN_DENY(synMemsetD16Async,(uint64_t a,const uint16_t value,const size_t n,const synStreamHandle s),(a,value,n,s))
SYN_DENY(synMemsetD8Async,(uint64_t a,const uint8_t value,const size_t n,const synStreamHandle s),(a,value,n,s))
SYN_DENY(synDeviceMalloc,(const synDeviceId d,const uint64_t n,uint64_t addr,const uint32_t flags,uint64_t*out),(d,n,addr,flags,out))
SYN_DENY(synDeviceFree,(const synDeviceId d,const uint64_t a,const uint32_t flags),(d,a,flags))
SYN_DENY(synHostMalloc,(const synDeviceId d,const uint64_t n,const uint32_t flags,void**out),(d,n,flags,out))
SYN_DENY(synHostFree,(const synDeviceId d,const void*a,const uint32_t flags),(d,a,flags))
SYN_DENY(synGraphCompile,(synRecipeHandle*r,const synGraphHandle g,const char*name,const char*log),(r,g,name,log))
#define HCCL_DENY(F,PARAMS,ARGS) extern "C" hcclResult_t F PARAMS {REAL(F);deny(#F);Guard guard;return real ARGS;}
HCCL_DENY(hcclReduce,(const void*a,void*b,size_t n,hcclDataType_t d,hcclRedOp_t op,int root,hcclComm_t c,void*s),(a,b,n,d,op,root,c,s))
HCCL_DENY(hcclBroadcast,(const void*a,void*b,size_t n,hcclDataType_t d,int root,hcclComm_t c,void*s),(a,b,n,d,root,c,s))
HCCL_DENY(hcclBcast,(void*b,size_t n,hcclDataType_t d,int root,hcclComm_t c,void*s),(b,n,d,root,c,s))
HCCL_DENY(hcclAlltoAll,(const void*a,void*b,size_t n,hcclDataType_t d,hcclComm_t c,void*s),(a,b,n,d,c,s))
HCCL_DENY(hcclBarrier,(hcclComm_t c,void*s),(c,s))
HCCL_DENY(hcclSend,(const void*a,size_t n,hcclDataType_t d,int peer,hcclComm_t c,void*s),(a,n,d,peer,c,s))
HCCL_DENY(hcclRecv,(void*b,size_t n,hcclDataType_t d,int peer,hcclComm_t c,void*s),(b,n,d,peer,c,s))
HCCL_DENY(hcclGroupStart,(),())
HCCL_DENY(hcclGroupEnd,(),())
