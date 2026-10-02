// Committing single-token entry. The scheduler owns request/page allocation;
// this plan owns pinned staging and a stream, and leases captured tensor storage.
// Shape/page changes are rejected before enqueue and handled by the caller.
struct NativeStepState {
  bool ready=false;
  bool padded_window=false;
  size_t position=SIZE_MAX,slot=SIZE_MAX,token=SIZE_MAX,usage=SIZE_MAX,window_usage=SIZE_MAX;
  size_t last_usage=0,last_window_usage=0,sample=SIZE_MAX;
  size_t list=SIZE_MAX,groups=SIZE_MAX,window_list=SIZE_MAX,window_groups=SIZE_MAX;
  int32_t pad_page=0;
  int32_t base_position=0,base_slot=0,last_position=0;
};
static NativeStepState native_step_state;
static synEventHandle step_begin{},step_end{};
static uint64_t last_device_span=0;
extern "C" int e1_last_device_span(uint64_t* out){
  if(replay_mode!=3||!last_device_span||!out)return -1;
  *out=last_device_span;return 0;
}

extern "C" int native_plan_prepare(int64_t* info) {
  if(!serial_stream||active.load()||rejected||native_step_state.ready)return -1;
  NativeStepState p;
  auto role=[&](const char* name,size_t bytes)->size_t {
    size_t found=SIZE_MAX;
    for(size_t i=0;i<commands.size();++i){auto& c=commands[i];
      if(c.semantic_role!=name)continue;
      if(found!=SIZE_MAX||c.kind!=6||c.type!=HOST_TO_DRAM||!c.count||
         (bytes&&c.count!=bytes)||c.host_bytes.size()!=c.count)return SIZE_MAX;
      found=i;
    }return found;
  };
  p.position=role("positions",4);p.slot=role("slot_mapping",4);p.token=role("token_ids",4);
  p.usage=role("block_usage",0);p.window_usage=role("window_block_usage",0);
  for(auto i:{p.position,p.slot,p.token,p.usage,p.window_usage})if(i==SIZE_MAX)return -2;
  std::memcpy(&p.base_position,commands[p.position].src,4);
  std::memcpy(&p.base_slot,commands[p.slot].src,4);
  if(p.base_position<0||p.base_slot<0||p.base_position%128<2||p.base_position%128>125||p.base_slot%128!=p.base_position%128)return -3;
  auto pages=[&](const char* bl,const char* bg,size_t usage,size_t& last)->bool {
    size_t list=role(bl,0),groups=role(bg,0);
    if(list==SIZE_MAX||groups==SIZE_MAX)return false;
    auto& l=commands[list];auto& g=commands[groups];auto& u=commands[usage];
    if(l.count%4||g.count!=l.count||u.count!=l.count/2)return false;
    bool found=false,padded=false;
    for(size_t i=0;i<l.count/4;++i){int32_t group,page;uint16_t actual;
      std::memcpy(&group,(char*)g.src+i*4,4);std::memcpy(&page,(char*)l.src+i*4,4);
      std::memcpy(&actual,(char*)u.src+i*2,2);
      if(group==-1){padded=true;continue;}
      if(group!=0||padded||page<0)return false;
      float value=float(page==p.base_slot/128?p.base_position%128+1:128);uint32_t bits;std::memcpy(&bits,&value,4);
      if(actual!=(bits>>16))return false;
      last=i;found=true;
    }
    if(!found)return false;
    int32_t page;std::memcpy(&page,(char*)l.src+last*4,4);return page==p.base_slot/128;
  };
  if(!pages("block_list","block_groups",p.usage,p.last_usage)||
     !pages("window_block_list","window_block_groups",p.window_usage,p.last_window_usage))return -4;
  p.list=role("block_list",0);p.groups=role("block_groups",0);
  p.window_list=role("window_block_list",0);p.window_groups=role("window_block_groups",0);
  if(p.last_usage+1<commands[p.list].count/4)
    std::memcpy(&p.pad_page,(char*)commands[p.list].src+4*(p.last_usage+1),4);
  for(size_t i=commands.size();i-->0;)if(commands[i].kind==6&&commands[i].type==DRAM_TO_HOST&&commands[i].count==4){p.sample=i;break;}
  if(p.sample==SIZE_MAX)return -5;
  p.last_position=p.base_position;p.ready=true;native_step_state=p;
  if(replay_mode==3){
    Guard guard;
    int rc=synEventCreate(&step_begin,device,EVENT_COLLECT_TIME);if(rc)return rc;
    rc=synEventCreate(&step_end,device,EVENT_COLLECT_TIME);if(rc)return rc;
  }
  info[0]=p.base_position;info[1]=p.base_slot;info[2]=commands.size();info[3]=launches;info[4]=collectives;
  return 0;
}

extern "C" int native_plan_capacities(int64_t* info){
  auto& p=native_step_state;if(!p.ready)return -1;
  info[0]=commands[p.list].count/4;info[1]=commands[p.window_list].count/4;
  return 0;
}

// Opt-in only after the caller certifies the qualified SWA consumer. Keep its
// captured metadata shape; unused pages have group=-1. No new buffer
// or pointer is introduced. The real cross-page replay gate is mandatory.
extern "C" int native_plan_enable_padded_window(){
  auto& p=native_step_state;
  if(!p.ready||active.load()||rejected||commands[p.window_list].count<8||
     commands[p.window_list].count%4||p.last_window_usage!=1)return -1;
  p.padded_window=true;
  return 0;
}

extern "C" int native_step(int32_t token,int32_t position,int32_t slot,int32_t* output,uint64_t* timing_out) {
  auto& p=native_step_state;
  // All failures here precede any mutation/submission.
  if(!p.ready||active.load()||rejected)return -1;
  if(position!=p.last_position+1||position/128!=p.base_position/128||
     (position%128==127&&(!p.padded_window||p.last_window_usage!=0))||
     slot!=p.base_slot+position-p.base_position)return -2;
  Guard guard;
  auto start=std::chrono::steady_clock::now();
  std::memcpy(commands[p.position].src,&position,4);std::memcpy(commands[p.slot].src,&slot,4);
  std::memcpy(commands[p.token].src,&token,4);
  float usage=float(position%128+1);uint32_t bits;std::memcpy(&bits,&usage,4);uint16_t bf16=bits>>16;
  std::memcpy((char*)commands[p.usage].src+2*p.last_usage,&bf16,2);
  std::memcpy((char*)commands[p.window_usage].src+2*p.last_window_usage,&bf16,2);
  uint64_t host=0,launch=0,network=0;
  if(replay_mode==3){int rc=synEventRecord(step_begin,serial_stream);if(rc)return rc;}
  int code=enqueue(1,&host,&launch,&network);
  if(!code&&replay_mode==3)code=synEventRecord(step_end,serial_stream);
  int sync_code=synStreamSynchronize(serial_stream);if(!code)code=sync_code;
  if(!code&&replay_mode==3)code=synEventElapsedTime(&last_device_span,step_begin,step_end);
  // Failed execution is fatal to this plan; it cannot silently fall back after
  // possibly submitting a prefix of the collective sequence.
  if(code){p.ready=false;return 10000+code;}
  std::memcpy(output,commands[p.sample].dst,4);
  p.last_position=position;
  timing_out[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  timing_out[1]=host;timing_out[2]=launch;timing_out[3]=network;
  return 0;
}

// Rebind non-contiguous paged-attention buffers without changing their shapes.
// Legacy plans skip offset127 through the bridge. Explicit padded-window
// plans retain their capacity and rebind one valid page at offset127.
extern "C" int native_step_bound(int32_t token,int32_t position,int32_t slot,
    const int32_t* pages,uint32_t count,const int32_t* window,uint32_t window_count,
    int32_t bridge_completed_position,int32_t* output,uint64_t* timing_out){
  auto& p=native_step_state;
  if(!p.ready||active.load()||rejected)return -1;
  bool bridge_boundary=(p.last_position%128==126&&position==p.last_position+2&&
                        bridge_completed_position==position-1&&position%128==0);
  if(position!=p.last_position+1&&!bridge_boundary)return -2;
  if(!pages)return native_step(token,position,slot,output,timing_out);
  uint32_t capacity=commands[p.list].count/4,wcapacity=commands[p.window_list].count/4;
  uint32_t expected_window=(p.padded_window&&position%128==127)?1:std::min<uint32_t>(2,count);
  if(position<0||(position%128==127&&!p.padded_window)||slot<0||slot%128!=position%128||!window||
     count!=uint32_t(position/128+1)||count>capacity||!window_count||window_count>wcapacity||
     window_count!=expected_window||pages[count-1]!=slot/128||window[window_count-1]!=slot/128)return -3;
  for(uint32_t i=0;i<count;++i)if(pages[i]<0)return -3;
  for(uint32_t i=0;i<window_count;++i)if(window[i]!=pages[count-window_count+i])return -3;
  // All validation precedes staging mutation and submission.
  auto fill=[&](size_t li,size_t gi,size_t ui,const int32_t* ids,uint32_t n,uint32_t cap){
    for(uint32_t j=0;j<cap;++j){
      int32_t page=j<n?ids[j]:p.pad_page,group=j<n?0:-1;
      float use=j<n?(j+1==n?float(position%128+1):128.f):1.f;
      uint32_t bits;std::memcpy(&bits,&use,4);uint16_t bf16=bits>>16;
      std::memcpy((char*)commands[li].src+j*4,&page,4);
      std::memcpy((char*)commands[gi].src+j*4,&group,4);
      std::memcpy((char*)commands[ui].src+j*2,&bf16,2);
    }
  };
  fill(p.list,p.groups,p.usage,pages,count,capacity);
  fill(p.window_list,p.window_groups,p.window_usage,window,window_count,wcapacity);
  p.last_usage=count-1;p.last_window_usage=window_count-1;
  p.base_position=position;p.base_slot=slot;
  if(bridge_boundary)p.last_position=bridge_completed_position;
  return native_step(token,position,slot,output,timing_out);
}

#include "native_multi.inc"

extern "C" int native_plan_reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if(active.load()||timing.recipe||profile_started||!boundary_commands.empty()||rollout.staging||!retired_commands.empty())return -1;
  Guard guard;
  if(serial_stream){int code=synStreamSynchronize(serial_stream);if(code)return code;
    for(auto* event:{&step_begin,&step_end})if(*event){
      code=synEventDestroy(*event);if(code)return code;*event=nullptr;
    }
    code=synStreamDestroy(serial_stream);if(code)return code;serial_stream=nullptr;}
  last_device_span=0;
  std::set<void*> buffers(owned_pinned_buffers.begin(),owned_pinned_buffers.end());
  for(auto& c:commands)for(auto* v:{&c.before,&c.after})for(auto& im:*v)if(im.host)buffers.insert(im.host);
  for(auto* pointer:buffers){int code=synHostFree(device,pointer,0);if(code)return code;}
  owned_pinned_buffers.clear();commands.clear();compact_commands.clear();recipe_metadata.clear();recorded_events.clear();
  metadata_queries=0;metadata_tensors=0;metadata_images=0;metadata_empty=0;metadata_omitted=0;partial_outputs=0;
  snapshot_bytes=0;launches=collectives=memcopies=external_events=external_waits=syncs=rejected=0;
  native_step_state=NativeStepState{};
  native_multi_state=NativeMultiState{};
  return 0;
}
