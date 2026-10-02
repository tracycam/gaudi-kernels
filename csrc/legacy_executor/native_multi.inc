// Named, fixed-shape input binding for multi-request / multi-query replay.
// The runner owns logical KV metadata; this entry owns pinned DMA staging.
struct NativeInputBinding {
  const char* role;
  const void* data;
  uint64_t bytes;
};
struct NativeMultiState {
  bool ready=false;
  std::map<std::string,size_t> inputs;
  size_t output=SIZE_MAX;
};
static NativeMultiState native_multi_state;
static const std::set<std::string> native_input_roles={
  "positions","slot_mapping","token_ids","logits_indices",
  "block_list","block_groups","block_usage",
  "window_block_list","window_block_groups","window_block_usage"
};

extern "C" int native_multi_prepare(uint64_t* output_bytes) {
  if(!output_bytes||!serial_stream||active.load()||rejected||
     native_multi_state.ready||native_step_state.ready)return -1;
  NativeMultiState candidate;
  for(size_t i=0;i<commands.size();++i){
    const auto& c=commands[i];
    if(c.semantic_role.empty())continue;
    if(c.kind!=6||!c.count)return -2;
    if(native_input_roles.count(c.semantic_role)){
      if(c.type!=HOST_TO_DRAM||!c.src||c.host_bytes.size()!=c.count||
         !candidate.inputs.emplace(c.semantic_role,i).second)return -3;
    }else if(c.semantic_role=="sampled_tokens"){
      if(c.type!=DRAM_TO_HOST||!c.dst||candidate.output!=SIZE_MAX)return -4;
      candidate.output=i;
    }else return -5;
  }
  if(candidate.inputs.size()!=native_input_roles.size()||candidate.output==SIZE_MAX)return -6;
  // Do not accept a captured prefix whose final host read precedes a launch.
  for(size_t i=candidate.output+1;i<commands.size();++i)
    if(commands[i].kind==1||commands[i].kind==2||commands[i].kind==3)return -7;
  candidate.ready=true;
  *output_bytes=commands[candidate.output].count;
  native_multi_state=std::move(candidate);
  return 0;
}

extern "C" int native_multi_input_bytes(const char* role,uint64_t* bytes){
  if(!native_multi_state.ready||!role||!bytes)return -1;
  auto it=native_multi_state.inputs.find(role);
  if(it==native_multi_state.inputs.end())return -2;
  *bytes=commands[it->second].count;return 0;
}

extern "C" int native_multi_read_output(void* output,uint64_t bytes){
  if(!native_multi_state.ready||!output||bytes!=commands[native_multi_state.output].count)return -1;
  std::memcpy(output,commands[native_multi_state.output].dst,bytes);return 0;
}

// Compare the runner's explicit dtype encoding with actual captured DMA bytes.
// This catches int64->int32 lowering or any unexpected bridge conversion.
extern "C" int native_multi_compare_input(const char* role,const void* data,uint64_t bytes){
  uint64_t expected=0;
  if(!data||native_multi_input_bytes(role,&expected)||expected!=bytes)return -1;
  return std::memcmp(commands[native_multi_state.inputs.at(role)].src,data,bytes)?-2:0;
}

extern "C" int native_multi_step(const NativeInputBinding* bindings,uint32_t count,
    void* output,uint64_t output_bytes,uint64_t* timing_out){
  auto& p=native_multi_state;
  if(!p.ready||active.load()||rejected||!bindings||!output||!timing_out)return -1;
  if(count!=p.inputs.size()||output_bytes!=commands[p.output].count)return -2;
  std::set<std::string> supplied;
  for(uint32_t i=0;i<count;++i){
    const auto& b=bindings[i];
    if(!b.role||!b.data||!supplied.insert(b.role).second)return -3;
    auto it=p.inputs.find(b.role);
    if(it==p.inputs.end()||b.bytes!=commands[it->second].count)return -4;
  }
  // No mutation or device submission until the whole input set is validated.
  Guard guard;
  auto start=std::chrono::steady_clock::now();
  for(uint32_t i=0;i<count;++i){
    const auto& b=bindings[i];
    std::memcpy(commands[p.inputs.at(b.role)].src,b.data,b.bytes);
  }
  uint64_t host=0,launch=0,network=0;
  int code=enqueue(1,&host,&launch,&network);
  int sync_code=synStreamSynchronize(serial_stream);if(!code)code=sync_code;
  // Submission errors are fatal. Never bridge-fallback after a partial replay.
  if(code){p.ready=false;return 10000+code;}
  std::memcpy(output,commands[p.output].dst,output_bytes);
  timing_out[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  timing_out[1]=host;timing_out[2]=launch;timing_out[3]=network;
  return 0;
}
