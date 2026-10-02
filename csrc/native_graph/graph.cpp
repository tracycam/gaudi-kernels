// SPDX-License-Identifier: Apache-2.0
#include "graph.h"
#include "ranges.hpp"
#include "capture_internal.h"
#include <synapse_api.h>
#include <hccl.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
struct Failure:std::runtime_error{gkg_status status;Failure(gkg_status s,const std::string&m):std::runtime_error(m),status(s){}};
void require(bool v,gkg_status s,const std::string&m){if(!v)throw Failure(s,m);}
void syn(synStatus s,const char*w){require(s==synSuccess,GKG_RUNTIME,std::string(w)+": Synapse "+std::to_string(int(s)));}
void hc(hcclResult_t s,const char*w){require(s==hcclSuccess,GKG_RUNTIME,std::string(w)+": HCCL "+std::to_string(int(s)));}
std::string named(const char*n){require(n&&*n,GKG_INVALID,"empty name");return n;}
bool external(gkg_role r){return r==GKG_EXTERNAL_READ||r==GKG_EXTERNAL_RW;}
uint64_t aligned(uint64_t x){require(x<=UINT64_MAX-255,GKG_RANGE,"alignment overflow");return(x+255)&~uint64_t(255);}
uint64_t plus(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,GKG_RANGE,"size overflow");return a+b;}
uint64_t times(uint64_t a,uint64_t b){require(a&&b&&a<=UINT64_MAX/b,GKG_RANGE,"size multiplication overflow");return a*b;}
unsigned bits(synDataType t){
 switch(t){case syn_type_single:case syn_type_int32:case syn_type_uint32:case syn_type_tf32:case syn_type_hb_float:return 32;
 case syn_type_bf16:case syn_type_fp16:case syn_type_int16:case syn_type_uint16:case syn_type_ufp16:return 16;
 case syn_type_int64:case syn_type_uint64:return 64;
 case syn_type_fixed:case syn_type_uint8:case syn_type_fp8_143:case syn_type_fp8_152:return 8;
 case syn_type_packed_mxfp4:case syn_type_packed_nf4:case syn_type_int4:case syn_type_uint4:return 4;
 default:throw Failure(GKG_UNSUPPORTED,"unregistered Synapse dtype");}
}
hcclDataType_t hc_type(gkg_dtype d){switch(d){case GKG_F32:return hcclFloat;case GKG_BF16:return hcclBfloat16;case GKG_I32:return hcclInt32;case GKG_U8:return hcclUint8;default:throw Failure(GKG_UNSUPPORTED,"unregistered HCCL dtype");}}
unsigned width(gkg_dtype d){switch(d){case GKG_F32:case GKG_I32:return 4;case GKG_BF16:return 2;case GKG_U8:return 1;default:throw Failure(GKG_UNSUPPORTED,"unregistered HCCL dtype");}}
std::string quoted(const std::string&s){std::string q="\"";for(unsigned char c:s){if(c=='\\'||c=='\"'){q+='\\';q+=c;}else if(c>=32&&c<127)q+=c;else{const char*hex="0123456789abcdef";q+="\\u00";q+=hex[c>>4];q+=hex[c&15];}}return q+'\"';}
struct Buffer{std::string name;uint64_t capture=0,bytes=0,address=0,pool_offset=0,host_offset=0;gkg_role role;};
struct Recipe{std::string name,path;synRecipeHandle handle{};uint64_t workspace=0;std::vector<synRetrievedLaunchTensorInfoExt> inventory;};
struct Comm{std::string name;hcclComm_t handle;int ranks=0,rank=0,declared_device=-1,queried_device=-1;};
struct Access{size_t buffer;uint64_t offset,bytes;bool write;};
struct Ref{size_t buffer;uint64_t offset;};
struct Command{
 int kind=0;size_t recipe=0,comm=0;gkg_collective collective{};gkg_dtype dtype{};uint64_t count=0,bytes=0;
 Ref src{},dst{};std::vector<Ref>refs;std::vector<synLaunchTensorInfoExt> bindings;
 std::vector<Access>access;std::vector<synSectionId>sections;std::vector<uint64_t>section_offsets;
};
struct Flight{synEventHandle event{};void*host=nullptr;uint64_t ticket=0;bool used=false,recorded=false;};
}
struct gkg_graph{
 std::mutex mutex;uint32_t device=0,capacity=0;std::string directory,error;bool ready=false,poisoned=false;
 uint64_t pool=0,pool_bytes=0,workspace_offset=0,workspace_bytes=0,host_bytes=0,sequence=0,replays=0;
 uint64_t supplied_pool_bytes=0;bool supplied_pool=false;
 synStreamHandle stream{};std::vector<Buffer>buffers;std::vector<Recipe>recipes;std::vector<Comm>comms;
 std::vector<Command>commands;std::vector<Flight>flights;uint64_t rejection_count=0;std::string capture_error;
 void building(){require(!ready&&!poisoned,GKG_STATE,"graph no longer building");}
 size_t buffer(const char*n){auto key=named(n);for(size_t i=0;i<buffers.size();++i)if(buffers[i].name==key)return i;throw Failure(GKG_INVALID,"undeclared buffer: "+key);}
 size_t recipe(const char*n){auto key=named(n);for(size_t i=0;i<recipes.size();++i)if(recipes[i].name==key)return i;throw Failure(GKG_INVALID,"undeclared recipe: "+key);}
 size_t comm(const char*n){auto key=named(n);for(size_t i=0;i<comms.size();++i)if(comms[i].name==key)return i;throw Failure(GKG_INVALID,"undeclared communicator: "+key);}
 Access access(size_t b,uint64_t offset,uint64_t bytes,bool write){require(gkg::fits(offset,bytes,buffers[b].bytes),GKG_RANGE,"access exceeds buffer: "+buffers[b].name);return{b,offset,bytes,write};}
 Flight&flight(uint64_t ticket){for(auto&f:flights)if(f.used&&f.ticket==ticket)return f;throw Failure(GKG_STALE,"unknown/released ticket");}
 void completed(Flight&f){require(f.recorded,GKG_RUNTIME,"terminal event not recorded; graph poisoned");auto s=synEventQuery(f.event);if(s==synBusy)throw Failure(GKG_BUSY,"flight incomplete");syn(s,"event query");}
};
template<class F>static gkg_status call(gkg_graph*g,F f){
 if(!g)return GKG_INVALID;
 std::lock_guard<std::mutex>lock(g->mutex);
 try{f();return GKG_OK;}catch(const Failure&e){g->error=e.what();++g->rejection_count;return e.status;}catch(const std::exception&e){g->error=e.what();++g->rejection_count;return GKG_RUNTIME;}
}
extern "C" gkg_status gkg_create(const gkg_config*c,gkg_graph**out){
 if(!c||!out||c->size!=sizeof(*c)||c->version!=GKG_ABI_VERSION||c->flights<1||c->flights>16||!c->artifact_directory)return GKG_INVALID;
 *out=nullptr;
 try{auto*g=new gkg_graph;g->device=c->device;g->capacity=c->flights;g->directory=c->artifact_directory;
  if(!std::filesystem::is_directory(g->directory)){delete g;return GKG_INVALID;}*out=g;return GKG_OK;
 }catch(...){return GKG_RUNTIME;}
}
extern "C" gkg_status gkg_buffer(gkg_graph*g,const char*n,uint64_t addr,uint64_t bytes,gkg_role role){return call(g,[&]{
 g->building();auto name=named(n);require(role>=GKG_INPUT&&role<=GKG_EXTERNAL_RW,GKG_INVALID,"invalid buffer role");
 require(bytes&&(!addr||gkg::valid_address(addr,bytes)),GKG_RANGE,"invalid buffer range");require(!external(role)||addr,GKG_INVALID,"external address required");
 for(auto&b:g->buffers){require(b.name!=name,GKG_INVALID,"duplicate buffer name");if(addr&&b.capture)require(addr>=b.capture+b.bytes||b.capture>=addr+bytes,GKG_RANGE,"overlapping buffers; register one base allocation and use offsets");}
 g->buffers.push_back({name,addr,bytes,external(role)?addr:0,0,0,role});
});}
extern "C" gkg_status gkg_output(gkg_graph*g,const char*n,uint64_t addr,uint64_t bytes){return call(g,[&]{
 g->building();auto name=named(n);require(gkg::valid_address(addr,bytes),GKG_RANGE,"invalid capture output range");
 for(auto&b:g->buffers)require(b.name!=name,GKG_INVALID,"duplicate output name");
 for(auto&b:g->buffers)if(b.capture==addr&&b.bytes==bytes){require(b.role==GKG_TEMP,GKG_INVALID,"only captured temporary can become output");b.name=name;b.role=GKG_OUTPUT;return;}
 for(auto&b:g->buffers)if(b.capture)require(addr>=b.capture+b.bytes||b.capture>=addr+bytes,GKG_RANGE,"output alias must match an exact captured base range");
 g->buffers.push_back({name,addr,bytes,0,0,0,GKG_OUTPUT});
});}
static void append_recipe(gkg_graph*g,const std::string&name,const std::string&path,synRecipeHandle handle){
 Recipe r;r.name=name;r.path=path;r.handle=handle;
 try{syn(synWorkspaceGetSize(&r.workspace,handle),"workspace size");uint32_t count=0;syn(synTensorRetrieveLaunchAmount(handle,&count),"launch inventory count");require(count&&count<=65536,GKG_RANGE,"invalid recipe inventory");
  std::vector<uint64_t>ids(count);syn(synTensorRetrieveLaunchIds(handle,ids.data(),count),"launch inventory ids");r.inventory.resize(count);for(size_t i=0;i<count;++i)r.inventory[i].tensorId=ids[i];
  syn(synTensorRetrieveLaunchInfoByIdExt(handle,count,r.inventory.data()),"launch inventory");
  for(auto&m:r.inventory)require(m.tensorType==DATA_TENSOR&&m.tensorDims&&m.tensorDims<=HABANA_DIM_MAX,GKG_UNSUPPORTED,"v1 supports static DATA_TENSOR bindings only");
  g->recipes.push_back(std::move(r));
 }catch(...){synRecipeDestroy(handle);throw;}
}
extern "C" gkg_status gkg_recipe(gkg_graph*g,const char*n,void*original){return call(g,[&]{
 g->building();auto name=named(n);require(original,GKG_INVALID,"null recipe");for(auto&r:g->recipes)require(r.name!=name,GKG_INVALID,"duplicate recipe name");
 auto path=g->directory+"/recipe-"+std::to_string(g->recipes.size())+".bin";require(!std::filesystem::exists(path),GKG_INVALID,"recipe output already exists");
 syn(synRecipeSerialize(static_cast<synRecipeHandle>(original),path.c_str()),"recipe serialize");synRecipeHandle clone{};syn(synRecipeDeSerialize(&clone,path.c_str()),"recipe clone");append_recipe(g,name,path,clone);
});}
extern "C" gkg_status gkg_recipe_load(gkg_graph*g,const char*n,const char*file){return call(g,[&]{
 g->building();auto name=named(n);for(auto&r:g->recipes)require(r.name!=name,GKG_INVALID,"duplicate recipe name");auto path=named(file);
 synRecipeHandle clone{};syn(synRecipeDeSerialize(&clone,path.c_str()),"recipe load");append_recipe(g,name,path,clone);
});}
extern "C" gkg_status gkg_launch(gkg_graph*g,const char*n,const gkg_binding*bindings,uint32_t count){return call(g,[&]{
 g->building();size_t rid=g->recipe(n);auto&r=g->recipes[rid];require(bindings&&count==r.inventory.size(),GKG_INVALID,"bindings must exactly cover compiled recipe inventory");
 Command c;c.kind=1;c.recipe=rid;std::vector<bool>seen(count,false);
 for(size_t i=0;i<count;++i){auto&b=bindings[i];require(b.size==sizeof(b)&&b.version==GKG_ABI_VERSION,GKG_INVALID,"binding ABI mismatch");auto tensor=named(b.tensor);size_t index=count;
  for(size_t j=0;j<count;++j)if(tensor==r.inventory[j].tensorName){index=j;break;}
  require(index<count&&!seen[index],GKG_INVALID,"duplicate/unknown tensor binding: "+tensor);seen[index]=true;auto&m=r.inventory[index];uint64_t bit_count=bits(m.tensorDataType);
  synLaunchTensorInfoExt launch{};launch.tensorId=m.tensorId;launch.tensorType=m.tensorType;
  for(uint32_t d=0;d<m.tensorDims;++d){require(m.tensorMinSize[d]==m.tensorMaxSize[d],GKG_UNSUPPORTED,"dynamic tensor: instantiate an explicitly fixed bucket");launch.tensorSize[d]=m.tensorMaxSize[d];bit_count=times(bit_count,m.tensorMaxSize[d]);}
  uint64_t bytes=plus(bit_count,7)/8;size_t bid=g->buffer(b.buffer);c.refs.push_back({bid,b.offset});c.bindings.push_back(launch);c.access.push_back(g->access(bid,b.offset,bytes,!m.isInput));c.sections.push_back(m.tensorSectionId);c.section_offsets.push_back(m.tensorOffsetInSection);
 }
 g->commands.push_back(std::move(c));
});}
static void register_comm(gkg_graph*g,const char*n,void*handle,int claimed){
 g->building();auto name=named(n);require(handle,GKG_INVALID,"null communicator");for(auto&c:g->comms)require(c.name!=name,GKG_INVALID,"duplicate communicator");Comm c{name,static_cast<hcclComm_t>(handle)};
 hc(hcclCommCount(c.handle,&c.ranks),"communicator size");hc(hcclCommUserRank(c.handle,&c.rank),"communicator rank");int device=-1;hc(hcclCommSynDevice(c.handle,&device),"communicator device");c.queried_device=device;
 require(device>=0||claimed>=0,GKG_UNSUPPORTED,"hcclCommSynDevice left output unchanged; explicit device ownership declaration required");c.declared_device=claimed>=0?claimed:device;
 require(c.declared_device==int(g->device)&&(device<0||device==int(g->device))&&c.ranks>0&&c.rank>=0&&c.rank<c.ranks,GKG_INVALID,"communicator/device mismatch: queried="+std::to_string(device)+" declared="+std::to_string(claimed)+" acquired="+std::to_string(g->device));g->comms.push_back(c);
}
extern "C" gkg_status gkg_comm(gkg_graph*g,const char*n,void*handle){return call(g,[&]{register_comm(g,n,handle,-1);});}
extern "C" gkg_status gkg_comm_on_device(gkg_graph*g,const char*n,void*handle,uint32_t device){return call(g,[&]{require(device<=INT32_MAX,GKG_INVALID,"invalid device declaration");register_comm(g,n,handle,int(device));});}
extern "C" gkg_status gkg_collect(gkg_graph*g,gkg_collective kind,const char*comm,const char*src,uint64_t so,const char*dst,uint64_t dso,uint64_t count,gkg_dtype dtype){return call(g,[&]{
 g->building();require(kind>=GKG_ALL_REDUCE&&kind<=GKG_REDUCE_SCATTER,GKG_UNSUPPORTED,"unregistered collective");Command c;c.kind=2;c.collective=kind;c.comm=g->comm(comm);c.src={g->buffer(src),so};c.dst={g->buffer(dst),dso};c.count=count;c.dtype=dtype;
 uint64_t bytes=times(count,width(dtype)),ranks=g->comms[c.comm].ranks;
 c.access.push_back(g->access(c.src.buffer,so,kind==GKG_REDUCE_SCATTER?times(bytes,ranks):bytes,false));c.access.push_back(g->access(c.dst.buffer,dso,kind==GKG_ALL_GATHER?times(bytes,ranks):bytes,true));g->commands.push_back(c);
});}
extern "C" gkg_status gkg_copy(gkg_graph*g,const char*src,uint64_t so,const char*dst,uint64_t dso,uint64_t bytes){return call(g,[&]{
 g->building();Command c;c.kind=3;c.src={g->buffer(src),so};c.dst={g->buffer(dst),dso};c.bytes=bytes;c.access={g->access(c.src.buffer,so,bytes,false),g->access(c.dst.buffer,dso,bytes,true)};
 require(c.src.buffer!=c.dst.buffer||so+bytes<=dso||dso+bytes<=so,GKG_RANGE,"overlapping D2D copy");g->commands.push_back(c);
});}
extern "C" gkg_status gkg_pool_requirements(gkg_graph*g,uint64_t*bytes){return call(g,[&]{
 g->building();require(bytes,GKG_INVALID,"null pool size output");uint64_t total=0,workspace=0;
 for(auto&b:g->buffers)if(!external(b.role))total=plus(aligned(total),b.bytes);
 for(auto&r:g->recipes)workspace=std::max(workspace,r.workspace);
 *bytes=plus(aligned(total),workspace);
});}
extern "C" gkg_status gkg_bind_pool(gkg_graph*g,uint64_t address,uint64_t bytes){return call(g,[&]{
 g->building();require(!g->supplied_pool,GKG_STATE,"dedicated pool already supplied");
 require(gkg::valid_address(address,bytes)&&address%256==0,GKG_RANGE,"dedicated pool must be nonempty and 256-byte aligned");
 g->pool=address;g->supplied_pool_bytes=bytes;g->supplied_pool=true;
});}
extern "C" gkg_status gkg_instantiate(gkg_graph*g){return call(g,[&]{
 g->building();require(!g->commands.empty(),GKG_INVALID,"empty plan");std::vector<gkg::Written>written(g->buffers.size());
 for(size_t i=0;i<g->buffers.size();++i)if(g->buffers[i].role==GKG_INPUT||external(g->buffers[i].role))written[i].add(0,g->buffers[i].bytes);
 for(size_t i=0;i<g->commands.size();++i){auto&c=g->commands[i];
  // Validate all reads before publishing any writes of this command. An
  // in-place output cannot justify reading uninitialized bytes of its input.
  for(auto&a:c.access)if(!a.write)require(written[a.buffer].contains(a.offset,a.bytes),GKG_UNINITIALIZED,"first read without producer: command "+std::to_string(i)+" buffer "+g->buffers[a.buffer].name+" offset "+std::to_string(a.offset));
  for(auto&a:c.access)if(a.write){require(g->buffers[a.buffer].role!=GKG_EXTERNAL_READ,GKG_READ_ONLY,"write to registered weight/read-only buffer");written[a.buffer].add(a.offset,a.bytes);}
 }
 for(size_t i=0;i<g->buffers.size();++i)if(g->buffers[i].role==GKG_OUTPUT)require(written[i].contains(0,g->buffers[i].bytes),GKG_UNINITIALIZED,"output not fully produced: "+g->buffers[i].name);
 for(auto&b:g->buffers)if(!external(b.role)){b.pool_offset=aligned(g->pool_bytes);g->pool_bytes=plus(b.pool_offset,b.bytes);}
 for(auto&r:g->recipes)g->workspace_bytes=std::max(g->workspace_bytes,r.workspace);
 g->workspace_offset=aligned(g->pool_bytes);g->pool_bytes=plus(g->workspace_offset,g->workspace_bytes);
 try{
  if(g->supplied_pool){
   require(g->supplied_pool_bytes>=g->pool_bytes,GKG_RANGE,"dedicated arena smaller than graph pool requirements");
   for(auto&b:g->buffers)if(b.capture)require(g->pool>=b.capture+b.bytes||b.capture>=g->pool+g->supplied_pool_bytes,GKG_RANGE,"dedicated arena aliases capture/external storage");
  }else if(g->pool_bytes)syn(synDeviceMalloc(g->device,g->pool_bytes,0,0,&g->pool),"private graph pool");
  for(auto&b:g->buffers)if(!external(b.role))b.address=plus(g->pool,b.pool_offset);
  for(auto&c:g->commands)if(c.kind==1){std::map<synSectionId,std::pair<uint64_t,uint64_t>>sections;
   for(size_t i=0;i<c.refs.size();++i){auto ref=c.refs[i];uint64_t address=plus(g->buffers[ref.buffer].address,ref.offset);c.bindings[i].pTensorAddress=address;require(address>=c.section_offsets[i],GKG_RANGE,"section offset underflow");uint64_t base=address-c.section_offsets[i];uint64_t size=plus(c.section_offsets[i],c.access[i].bytes);auto inserted=sections.emplace(c.sections[i],std::make_pair(base,size));require(inserted.second||inserted.first->second.first==base,GKG_RANGE,"recipe section aliases require a common registered base buffer");inserted.first->second.second=std::max(inserted.first->second.second,size);}
   for(auto a=sections.begin();a!=sections.end();++a)for(auto b=std::next(a);b!=sections.end();++b)require(a->second.first+a->second.second<=b->second.first||b->second.first+b->second.second<=a->second.first,GKG_RANGE,"different persistent recipe sections overlap; use disjoint buffers");
  }
  for(auto&b:g->buffers)if(b.role==GKG_INPUT||b.role==GKG_OUTPUT){b.host_offset=aligned(g->host_bytes);g->host_bytes=plus(b.host_offset,b.bytes);}
  syn(synStreamCreateGeneric(&g->stream,g->device,0),"owned replay stream");g->flights.resize(g->capacity);
  for(auto&f:g->flights){syn(synEventCreate(&f.event,g->device,0),"owned terminal event");if(g->host_bytes)syn(synHostMalloc(g->device,g->host_bytes,0,&f.host),"owned flight staging");}
  g->ready=true;
 }catch(...){g->poisoned=true;throw;}
});}
extern "C" gkg_status gkg_replay(gkg_graph*g,const gkg_input*inputs,uint32_t count,uint64_t*ticket){return call(g,[&]{
 require(g->ready&&!g->poisoned,GKG_STATE,"graph not ready/poisoned");require(ticket&&(!count||inputs),GKG_INVALID,"null replay input/ticket");size_t required=0;
 for(auto&b:g->buffers)if(b.role==GKG_INPUT){++required;int found=0;for(uint32_t i=0;i<count;++i)if(inputs[i].name&&b.name==inputs[i].name){++found;require(inputs[i].size==sizeof(gkg_input)&&inputs[i].version==GKG_ABI_VERSION&&inputs[i].data&&inputs[i].bytes==b.bytes,GKG_INVALID,"input ABI/size mismatch: "+b.name);}require(found==1,GKG_INVALID,"missing/duplicate input: "+b.name);}
 require(count==required,GKG_INVALID,"unknown replay input");Flight*f=nullptr;for(auto&slot:g->flights)if(!slot.used){f=&slot;break;}require(f,GKG_BUSY,"all flight slots occupied; release a completed ticket");
 f->used=true;f->recorded=false;f->ticket=++g->sequence;*ticket=f->ticket;
 try{
  for(auto&b:g->buffers)if(b.role==GKG_INPUT){const void*source=nullptr;for(uint32_t i=0;i<count;++i)if(b.name==inputs[i].name)source=inputs[i].data;auto*host=static_cast<char*>(f->host)+b.host_offset;std::memcpy(host,source,b.bytes);syn(synMemCopyAsync(g->stream,reinterpret_cast<uint64_t>(host),b.bytes,b.address,HOST_TO_DRAM),"static input upload");}
  for(auto&c:g->commands){
   if(c.kind==1)syn(synLaunchWithExternalEventsExt(g->stream,c.bindings.data(),c.bindings.size(),g->workspace_bytes?g->pool+g->workspace_offset:0,g->recipes[c.recipe].handle,nullptr,0,0),"native recipe launch");
   else{auto src=reinterpret_cast<void*>(g->buffers[c.src.buffer].address+c.src.offset);auto dst=reinterpret_cast<void*>(g->buffers[c.dst.buffer].address+c.dst.offset);
    if(c.kind==3)syn(synMemCopyAsync(g->stream,reinterpret_cast<uint64_t>(src),c.bytes,reinterpret_cast<uint64_t>(dst),DRAM_TO_DRAM),"native D2D copy");
    else{auto comm=g->comms[c.comm].handle;auto dtype=hc_type(c.dtype);
     if(c.collective==GKG_ALL_REDUCE)hc(hcclAllReduce(src,dst,c.count,dtype,hcclSum,comm,g->stream),"native allreduce");
     else if(c.collective==GKG_ALL_GATHER)hc(hcclAllGather(src,dst,c.count,dtype,comm,g->stream),"native allgather");
     else hc(hcclReduceScatter(src,dst,c.count,dtype,hcclSum,comm,g->stream),"native reduce scatter");
    }
   }
  }
  for(auto&b:g->buffers)if(b.role==GKG_OUTPUT)syn(synMemCopyAsync(g->stream,b.address,b.bytes,reinterpret_cast<uint64_t>(static_cast<char*>(f->host)+b.host_offset),DRAM_TO_HOST),"static output download");
  syn(synEventRecord(f->event,g->stream),"terminal record");f->recorded=true;++g->replays;
 }catch(...){g->poisoned=true;f->recorded=synEventRecord(f->event,g->stream)==synSuccess;throw;}
});}
extern "C" gkg_status gkg_query(gkg_graph*g,uint64_t t){return call(g,[&]{g->completed(g->flight(t));});}
extern "C" gkg_status gkg_wait(gkg_graph*g,uint64_t t){return call(g,[&]{auto&f=g->flight(t);require(f.recorded,GKG_RUNTIME,"missing terminal event");syn(synEventSynchronize(f.event),"flight event wait");});}
extern "C" gkg_status gkg_read(gkg_graph*g,uint64_t t,const char*n,void*dst,uint64_t bytes){return call(g,[&]{auto&f=g->flight(t);g->completed(f);auto&b=g->buffers[g->buffer(n)];require(b.role==GKG_OUTPUT&&dst&&bytes==b.bytes,GKG_INVALID,"output role/size mismatch");std::memcpy(dst,static_cast<char*>(f.host)+b.host_offset,bytes);});}
extern "C" gkg_status gkg_release(gkg_graph*g,uint64_t t){return call(g,[&]{auto&f=g->flight(t);g->completed(f);f.used=false;});}
extern "C" gkg_status gkg_check_external(gkg_graph*g,const char*n,uint64_t address,uint64_t bytes){return call(g,[&]{auto&b=g->buffers[g->buffer(n)];require(external(b.role)&&b.address==address&&b.bytes==bytes,GKG_STALE,"external moved/resized: "+b.name);});}
extern "C" gkg_status gkg_address(gkg_graph*g,const char*n,uint64_t*addr){return call(g,[&]{require(g->ready&&addr,GKG_STATE,"graph not ready/address null");*addr=g->buffers[g->buffer(n)].address;});}
extern "C" gkg_status gkg_stream(gkg_graph*g,void**stream){return call(g,[&]{require(g->ready&&stream,GKG_STATE,"graph not ready/stream null");*stream=g->stream;});}
extern "C" gkg_status gkg_capture_address(gkg_graph*g,uint64_t address,uint64_t bytes,int write,char*name,uint32_t capacity,uint64_t*offset){return call(g,[&]{
 g->building();require(name&&offset&&gkg::valid_address(address,bytes),GKG_INVALID,"capture address invalid");
 for(auto&b:g->buffers)if(b.capture&&address>=b.capture&&gkg::fits(address-b.capture,bytes,b.bytes)){
  require(capacity>b.name.size(),GKG_RANGE,"capture name buffer too small");std::strcpy(name,b.name.c_str());*offset=address-b.capture;return;
 }
 require(write,GKG_UNINITIALIZED,"unclassified first-read capture address: "+std::to_string(address)+" bytes "+std::to_string(bytes));
 for(auto&b:g->buffers)if(b.capture)require(address>=b.capture+b.bytes||b.capture>=address+bytes,GKG_RANGE,"ambiguous capture alias; declare its complete base allocation");
 auto n="capture-output-"+std::to_string(g->buffers.size());require(capacity>n.size(),GKG_RANGE,"capture name buffer too small");g->buffers.push_back({n,address,bytes,0,0,0,GKG_TEMP});std::strcpy(name,n.c_str());*offset=0;
});}
extern "C" gkg_status gkg_capture_communicator(gkg_graph*g,void*handle,char*name,uint32_t capacity){return call(g,[&]{
 g->building();for(auto&c:g->comms)if(c.handle==handle){require(name&&capacity>c.name.size(),GKG_RANGE,"communicator name buffer too small");std::strcpy(name,c.name.c_str());return;}throw Failure(GKG_UNSUPPORTED,"unregistered captured HCCL communicator");
});}
extern "C" gkg_status gkg_capture_fail(gkg_graph*g,gkg_status status,const char*message){return call(g,[&]{g->poisoned=true;g->capture_error=message?message:"capture rejected";throw Failure(status,g->capture_error);});}
extern "C" const char*gkg_error(gkg_graph*g){return g?g->error.c_str():"invalid graph";}
extern "C" gkg_status gkg_report(gkg_graph*g,const char*file){return call(g,[&]{
 std::ofstream out(named(file));require(bool(out),GKG_INVALID,"cannot open report");char driver[256]{};auto driver_status=synDriverGetVersion(driver,sizeof(driver));
 out<<"{\"abi_version\":1,\"device\":"<<g->device<<",\"ready\":"<<(g->ready?"true":"false")<<",\"poisoned\":"<<(g->poisoned?"true":"false")<<",\"driver\":"<<quoted(driver_status==synSuccess?driver:"unavailable")<<",\"commands\":"<<g->commands.size()<<",\"replays\":"<<g->replays<<",\"rejections\":"<<g->rejection_count<<",\"last_error\":"<<quoted(g->error)<<",\"pool_bytes\":"<<g->pool_bytes<<",\"pool_source\":"<<quoted(g->supplied_pool?"retained dedicated producer arena":"Synapse owned allocation")<<",\"workspace_bytes\":"<<g->workspace_bytes<<",\"flight_capacity\":"<<g->capacity<<",\"buffers\":[";
 for(size_t i=0;i<g->buffers.size();++i){auto&b=g->buffers[i];if(i)out<<',';out<<"{\"name\":"<<quoted(b.name)<<",\"role\":"<<int(b.role)<<",\"capture_address\":"<<b.capture<<",\"replay_address\":"<<b.address<<",\"bytes\":"<<b.bytes<<'}';}out<<"],\"recipes\":[";
 for(size_t i=0;i<g->recipes.size();++i){auto&r=g->recipes[i];if(i)out<<',';out<<"{\"name\":"<<quoted(r.name)<<",\"file\":"<<quoted(r.path)<<",\"workspace_bytes\":"<<r.workspace<<",\"launch_tensors\":"<<r.inventory.size()<<'}';}out<<"],\"communicators\":[";
 for(size_t i=0;i<g->comms.size();++i){auto&c=g->comms[i];if(i)out<<',';out<<"{\"name\":"<<quoted(c.name)<<",\"ranks\":"<<c.ranks<<",\"rank\":"<<c.rank<<",\"declared_device\":"<<c.declared_device<<",\"queried_device\":"<<c.queried_device<<'}';}out<<"],\"plan_commands\":[";
 for(size_t i=0;i<g->commands.size();++i){auto&c=g->commands[i];if(i)out<<',';
  if(c.kind==1){auto&r=g->recipes[c.recipe];out<<"{\"kind\":\"launch\",\"recipe\":"<<quoted(r.name)<<",\"bindings\":{";
   for(size_t j=0;j<c.refs.size();++j){if(j)out<<',';const char*name=nullptr;for(auto&m:r.inventory)if(m.tensorId==c.bindings[j].tensorId)name=m.tensorName;require(name,GKG_RUNTIME,"report tensor id missing");out<<quoted(name)<<":["<<quoted(g->buffers[c.refs[j].buffer].name)<<','<<c.refs[j].offset<<']';}out<<"}}";
  }else{out<<"{\"kind\":"<<quoted(c.kind==2?"collect":"copy")<<",\"src\":"<<quoted(g->buffers[c.src.buffer].name)<<",\"dst\":"<<quoted(g->buffers[c.dst.buffer].name)<<",\"src_offset\":"<<c.src.offset<<",\"dst_offset\":"<<c.dst.offset;
   if(c.kind==2)out<<",\"collective\":"<<int(c.collective)<<",\"comm\":"<<quoted(g->comms[c.comm].name)<<",\"count\":"<<c.count<<",\"dtype\":"<<int(c.dtype);
   else out<<",\"bytes\":"<<c.bytes;
   out<<'}';
  }
 }out<<"],\"capture_error\":"<<quoted(g->capture_error)<<",\"coverage\":{\"recipe_launches\":"<<std::count_if(g->commands.begin(),g->commands.end(),[](auto&c){return c.kind==1;})<<",\"collectives\":"<<std::count_if(g->commands.begin(),g->commands.end(),[](auto&c){return c.kind==2;})<<",\"d2d_copies\":"<<std::count_if(g->commands.begin(),g->commands.end(),[](auto&c){return c.kind==3;})<<"},\"scope\":\"public API C++ replay; not one mixed hardware submission\"}\n";
 require(bool(out),GKG_RUNTIME,"report write failed");
});}
extern "C" gkg_status gkg_destroy(gkg_graph*g){
 if(!g)return GKG_INVALID;
 std::unique_lock<std::mutex>lock(g->mutex);
 for(auto&f:g->flights)if(f.used){g->error="outstanding flight ticket";return GKG_BUSY;}
 // No work remains after all tickets were event-completed and released.
 gkg_status status=GKG_OK;
 for(auto&f:g->flights){if(f.host&&synHostFree(g->device,f.host,0)!=synSuccess)status=GKG_RUNTIME;if(f.event&&synEventDestroy(f.event)!=synSuccess)status=GKG_RUNTIME;}
 if(g->stream&&synStreamDestroy(g->stream)!=synSuccess)status=GKG_RUNTIME;
 if(g->pool&&!g->supplied_pool&&synDeviceFree(g->device,g->pool,0)!=synSuccess)status=GKG_RUNTIME;
 for(auto&r:g->recipes)if(synRecipeDestroy(r.handle)!=synSuccess)status=GKG_RUNTIME;
 lock.unlock();delete g;return status;
}
