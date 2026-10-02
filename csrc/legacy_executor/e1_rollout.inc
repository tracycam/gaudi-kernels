// Experimental autoregressive rollout restricted to the current KV page at any validated context.
// This is an oracle-comparison fixture, not the persistent production executor:
// restore the captured feedback scalars afterward so the bridge can validate it.
struct RolloutState {
  size_t position=SIZE_MAX,slot=SIZE_MAX,usage=SIZE_MAX,window_usage=SIZE_MAX,sample=SIZE_MAX;
  uint64_t feedback=0,sample_device=0;
  int32_t base_position=0,base_slot=0;
  unsigned capacity=0;
  size_t stride=0,usage_offset=8,window_usage_offset=0,usage_last=0,window_usage_last=0;
  unsigned char* staging=nullptr;
  int32_t* outputs=nullptr;
  int32_t* restore=nullptr;
  std::array<void*,4> original_sources{};
  void* original_sample_destination=nullptr;
};
static RolloutState rollout;
static uint64_t rollout_thread_cpu_ns=0;
extern "C" uint64_t e1_rollout_thread_cpu_ns(){return rollout_thread_cpu_ns;}

extern "C" int e1_rollout_init(uint64_t feedback,unsigned capacity) {
  if(!serial_stream||rejected||!capacity||rollout.staging)return -1;
  // Validation must use a real, explicitly identified live scalar. The pair
  // of mutations restores it and also rejects a writable/unknown allocation.
  int checked=e1_perturb_live_input(feedback);
  if(checked<0)return -2;
  if(e1_perturb_live_input(feedback)!=checked)return -3;
  auto role=[&](const char* name,size_t bytes)->size_t {
    size_t found=SIZE_MAX;
    for(size_t i=0;i<commands.size();++i){const auto& c=commands[i];
      if(c.semantic_role!=name)continue;
      if(found!=SIZE_MAX||c.kind!=6||c.type!=HOST_TO_DRAM||(bytes&&c.count!=bytes)||c.host_bytes.size()!=c.count)return SIZE_MAX;
      found=i;
    }
    return found;
  };
  rollout.position=role("positions",4);rollout.slot=role("slot_mapping",4);
  rollout.usage=role("block_usage",0);rollout.window_usage=role("window_block_usage",0);
  auto blocks=role("block_list",0),window=role("window_block_list",0);
  auto groups=role("block_groups",0),window_groups=role("window_block_groups",0);
  for(auto i:{rollout.position,rollout.slot,rollout.usage,rollout.window_usage,blocks,window,groups,window_groups})
    if(i==SIZE_MAX)return -4;
  auto i32=[&](size_t i){int32_t v;std::memcpy(&v,commands[i].host_bytes.data(),4);return v;};
  rollout.base_position=i32(rollout.position);rollout.base_slot=i32(rollout.slot);
  if(rollout.base_position<0||unsigned(rollout.base_position%128)+capacity>=127)return -5;
  if(rollout.base_slot<0||rollout.base_slot%128!=rollout.base_position%128)return -6;
  // Only the final active page usage changes. Padding and all page IDs stay
  // byte-identical. Reject page/SWA block-list transitions, including offset127.
  auto validate_pages=[&](size_t bi,size_t gi,size_t ui,size_t& last,std::vector<int32_t>& pages)->bool {
    auto& bc=commands[bi];auto& gc=commands[gi];auto& uc=commands[ui];
    if(!bc.count||bc.count%4||gc.count!=bc.count||uc.count!=bc.count/2)return false;
    bool padding=false;size_t count=bc.count/4;
    for(size_t n=0;n<count;++n){
      int32_t group,page;uint16_t usage;
      std::memcpy(&group,gc.host_bytes.data()+4*n,4);
      std::memcpy(&page,bc.host_bytes.data()+4*n,4);
      std::memcpy(&usage,uc.host_bytes.data()+2*n,2);
      if(group==-1){padding=true;continue;}
      if(group!=0||padding||page<0)return false;
      pages.push_back(page);last=n;
      float expected=float(page==rollout.base_slot/128?rollout.base_position%128+1:128);
      uint32_t bits;std::memcpy(&bits,&expected,4);
      if(usage!=uint16_t(bits>>16))return false;
    }
    return !pages.empty()&&pages.back()==rollout.base_slot/128;
  };
  std::vector<int32_t> full_pages,window_pages;
  if(!validate_pages(blocks,groups,rollout.usage,rollout.usage_last,full_pages)||
     !validate_pages(window,window_groups,rollout.window_usage,rollout.window_usage_last,window_pages))return -6;
  if(full_pages.size()!=size_t(rollout.base_position/128+1)||
     window_pages.size()!=std::min<size_t>(2,full_pages.size())||
     !std::equal(window_pages.begin(),window_pages.end(),full_pages.end()-window_pages.size()))return -6;
  rollout.window_usage_offset=rollout.usage_offset+commands[rollout.usage].count;
  rollout.stride=(rollout.window_usage_offset+commands[rollout.window_usage].count+63)&~size_t(63);
  for(size_t i=commands.size();i-->0;)if(commands[i].kind==6&&commands[i].type==DRAM_TO_HOST&&commands[i].count==4){rollout.sample=i;break;}
  if(rollout.sample==SIZE_MAX)return -7;
  rollout.feedback=feedback;rollout.sample_device=(uint64_t)commands[rollout.sample].src;
  if(feedback==rollout.sample_device)return -8;
  rollout.capacity=capacity;
  Guard guard;
  if(synHostMalloc(device,capacity*rollout.stride,0,(void**)&rollout.staging)!=synSuccess||
     synHostMalloc(device,capacity*4,0,(void**)&rollout.outputs)!=synSuccess||
     synHostMalloc(device,8,0,(void**)&rollout.restore)!=synSuccess)return -9;
  if(synMemCopyAsync(serial_stream,feedback,4,(uint64_t)rollout.restore,DRAM_TO_HOST)!=synSuccess||
     synMemCopyAsync(serial_stream,rollout.sample_device,4,(uint64_t)(rollout.restore+1),DRAM_TO_HOST)!=synSuccess||
     synStreamSynchronize(serial_stream)!=synSuccess)return -10;
  unsigned j=0;for(auto i:{rollout.position,rollout.slot,rollout.usage,rollout.window_usage})rollout.original_sources[j++]=commands[i].src;
  rollout.original_sample_destination=commands[rollout.sample].dst;
  return 0;
}

extern "C" int e1_rollout_run(unsigned steps,int32_t* tokens,uint64_t* timing) {
  if(!rollout.staging||!steps||steps>rollout.capacity)return -1;
  Guard guard;
  int code=synStreamSynchronize(serial_stream);if(code)return code;
  auto start=std::chrono::steady_clock::now();
  struct timespec cpu_start{},cpu_end{};
  bool cpu_clock_valid=clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_start)==0;
  rollout_thread_cpu_ns=0;
  // Independent pinned sources for every queued token prevent host lookahead
  // from overwriting the metadata of DMA commands still pending on the device.
  for(unsigned j=0;j<steps;++j){
    int32_t position=rollout.base_position+j+1,slot=rollout.base_slot+j+1;
    float usage=float(position%128+1);uint32_t usage_bits;std::memcpy(&usage_bits,&usage,4);
    uint16_t bf16=uint16_t(usage_bits>>16);
    auto* p=rollout.staging+rollout.stride*j;
    std::memcpy(p,&position,4);std::memcpy(p+4,&slot,4);
    std::memcpy(p+rollout.usage_offset,commands[rollout.usage].host_bytes.data(),commands[rollout.usage].count);
    std::memcpy(p+rollout.window_usage_offset,commands[rollout.window_usage].host_bytes.data(),commands[rollout.window_usage].count);
    std::memcpy(p+rollout.usage_offset+2*rollout.usage_last,&bf16,2);
    std::memcpy(p+rollout.window_usage_offset+2*rollout.window_usage_last,&bf16,2);
  }
  uint64_t host=0,launch=0,hccl=0;
  for(unsigned j=0;j<steps;++j){
    auto* p=rollout.staging+rollout.stride*j;
    commands[rollout.position].src=p;commands[rollout.slot].src=p+4;
    commands[rollout.usage].src=p+rollout.usage_offset;commands[rollout.window_usage].src=p+rollout.window_usage_offset;
    commands[rollout.sample].dst=rollout.outputs+j;
    code=synMemCopyAsync(serial_stream,rollout.sample_device,4,rollout.feedback,DRAM_TO_DRAM);
    if(code)break;
    uint64_t h=0,l=0,c=0;code=enqueue(1,&h,&l,&c);host+=h;launch+=l;hccl+=c;
    if(code)break;
  }
  int sync_code=synStreamSynchronize(serial_stream);if(!code)code=sync_code;
  auto end=std::chrono::steady_clock::now();
  if(cpu_clock_valid&&clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_end)==0)
    rollout_thread_cpu_ns=uint64_t((cpu_end.tv_sec-cpu_start.tv_sec)*1000000000LL+cpu_end.tv_nsec-cpu_start.tv_nsec);
  timing[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  timing[1]=host;timing[2]=launch;timing[3]=hccl;
  if(!code)std::memcpy(tokens,rollout.outputs,steps*4);
  unsigned j=0;for(auto i:{rollout.position,rollout.slot,rollout.usage,rollout.window_usage})commands[i].src=rollout.original_sources[j++];
  commands[rollout.sample].dst=rollout.original_sample_destination;
  // Future KV rows remain written, but causal masks exclude them in the
  // subsequent bridge rollout. Its token-by-token oracle checks this claim.
  int restore_code=synMemCopyAsync(serial_stream,(uint64_t)rollout.restore,4,rollout.feedback,HOST_TO_DRAM);
  if(!restore_code)restore_code=synMemCopyAsync(serial_stream,(uint64_t)(rollout.restore+1),4,rollout.sample_device,HOST_TO_DRAM);
  if(!restore_code)restore_code=synStreamSynchronize(serial_stream);
  return code?code:restore_code;
}
