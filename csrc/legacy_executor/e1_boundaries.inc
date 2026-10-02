// Source-level operator inventory, followed by device event boundaries. The
// paired uninstrumented trial quantifies event-recording overhead.
struct RecipeCatalogEntry { std::string name;std::map<std::string,unsigned> guids; };
static std::mutex catalog_mutex;
static std::map<void*,std::map<std::string,unsigned>> graph_catalog;
static std::map<void*,RecipeCatalogEntry> recipe_catalog;
static bool boundaries_enabled(){return option(64);}
extern "C" synStatus synGraphCreate(synGraphHandle* graph,const synDeviceType type){
  REAL(synGraphCreate);auto status=real(graph,type);
  if(status==synSuccess&&boundaries_enabled()){std::lock_guard<std::mutex> lock(catalog_mutex);graph_catalog[(void*)*graph].clear();}
  return status;
}
extern "C" synStatus synGraphDestroy(const synGraphHandle graph){
  REAL(synGraphDestroy);auto status=real(graph);
  if(status==synSuccess&&boundaries_enabled()){std::lock_guard<std::mutex> lock(catalog_mutex);graph_catalog.erase((void*)graph);}
  return status;
}
extern "C" synStatus synNodeCreateWithId(const synGraphHandle graph,const synTensor* inputs,const synTensor* outputs,
    const uint32_t ni,const uint32_t no,const void* params,const unsigned size,const char* guid,const char* name,
    synNodeId* id,const char** ilayouts,const char** olayouts){
  REAL(synNodeCreateWithId);auto status=real(graph,inputs,outputs,ni,no,params,size,guid,name,id,ilayouts,olayouts);
  if(status==synSuccess&&boundaries_enabled()){std::lock_guard<std::mutex> lock(catalog_mutex);++graph_catalog[(void*)graph][guid?guid:"<null>"];}
  return status;
}
extern "C" synStatus synGraphCompile(synRecipeHandle* recipe,const synGraphHandle graph,const char* name,const char* log){
  REAL(synGraphCompile);auto status=real(recipe,graph,name,log);
  if(status==synSuccess&&boundaries_enabled()){
    std::lock_guard<std::mutex> lock(catalog_mutex);recipe_catalog[(void*)*recipe]={name?name:"",graph_catalog[(void*)graph]};
  }
  return status;
}
static std::vector<size_t> boundary_commands,boundary_markers;
static std::vector<synEventHandle> boundary_events;
static std::string csv_field(const std::string& in){std::string s="\"";for(char c:in){if(c=='\"')s+='\"';s+=c;}return s+'\"';}
extern "C" int e1_boundary_init(const char* manifest,unsigned reserve){
  if(!timing.recipe||!boundary_commands.empty())return -1;
  for(size_t i=0;i<commands.size();++i)if(commands[i].kind==1||commands[i].kind==2||commands[i].kind==3||commands[i].kind==6||commands[i].kind==7)boundary_commands.push_back(i);
  for(auto i:boundary_commands)if(commands[i].kind==1||commands[i].kind==2||commands[i].kind==3)boundary_markers.push_back(i);
  if(boundary_markers.empty()||boundary_markers.back()!=boundary_commands.back())boundary_markers.push_back(boundary_commands.back());
  boundary_events.resize(std::max<size_t>(reserve,boundary_markers.size()+1));
  Guard guard;
  for(auto& event:boundary_events)if(synEventCreate(&event,device,EVENT_COLLECT_TIME)!=synSuccess)return -2;
  FILE* out=fopen(manifest,"w");if(!out)return -3;
  fprintf(out,"stage,command,kind,count,type,recipe,role,recipe_name,source_guids\n");
  for(size_t j=0;j<boundary_markers.size();++j){auto i=boundary_markers[j];const auto& c=commands[i];
    std::string guids,name;
    if(c.kind==1){const auto found=recipe_catalog.find(c.object);if(found!=recipe_catalog.end()){
      name=found->second.name;
      for(const auto& pair:found->second.guids)guids+=pair.first+":"+std::to_string(pair.second)+";";
    }}
    fprintf(out,"%zu,%zu,%d,%llu,%d,%llu,%s,%s,%s\n",j,i,c.kind,(unsigned long long)c.count,c.type,(unsigned long long)c.object,
      csv_field(c.semantic_role).c_str(),csv_field(name).c_str(),csv_field(guids).c_str());
  }
  fclose(out);return (int)boundary_markers.size();
}
extern "C" int e1_boundary_trial(uint64_t* out,unsigned count){
  if(count<boundary_markers.size()+3||boundary_events.empty())return -1;
  Guard guard;
  auto launch=reinterpret_cast<decltype(&synLaunchWithExternalEventsExt)>(resolve("synLaunchWithExternalEventsExt"));
  auto ar=reinterpret_cast<decltype(&hcclAllReduce)>(resolve("hcclAllReduce"));
  auto ag=reinterpret_cast<decltype(&hcclAllGather)>(resolve("hcclAllGather"));
  auto copy=reinterpret_cast<decltype(&synMemCopyAsync)>(resolve("synMemCopyAsync"));
  auto copies=reinterpret_cast<decltype(&synMemCopyAsyncMultiple)>(resolve("synMemCopyAsyncMultiple"));
  int code=synStreamSynchronize(serial_stream);if(code)return code;
  code=synLaunch(serial_stream,timing.binding,2,timing.workspace,timing.recipe,0);if(code)return code;
  code=align_timing_gate();if(code)return code;
  code=synEventRecord(boundary_events[0],serial_stream);if(code)return code;
  auto start=std::chrono::steady_clock::now();
  size_t marker=0;
  for(size_t j=0;j<boundary_commands.size();++j){auto& c=commands[boundary_commands[j]];
    if(c.kind==1)code=launch(serial_stream,c.tensors.data(),c.tensors.size(),c.workspace,(synRecipeHandle)c.object,nullptr,0,c.flags);
    else if(c.kind==2)code=ar(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclRedOp_t)c.op,(hcclComm_t)c.object,serial_stream);
    else if(c.kind==3)code=ag(c.src,c.dst,c.count,(hcclDataType_t)c.type,(hcclComm_t)c.object,serial_stream);
    else if(c.kind==6)code=copy(serial_stream,(uint64_t)c.src,c.count,(uint64_t)c.dst,(synDmaDir)c.type);
    else if(c.kind==7)code=copies(serial_stream,c.multi_src.data(),c.multi_size.data(),c.multi_dst.data(),(synDmaDir)c.type,c.multi_src.size());
    if(code)return code;
    if(marker<boundary_markers.size()&&boundary_commands[j]==boundary_markers[marker]){
      code=synEventRecord(boundary_events[++marker],serial_stream);if(code)return code;
    }
  }
  out[2]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  out[1]=synEventQuery(boundary_events[0]);
  code=synStreamSynchronize(serial_stream);if(code)return code;
  code=synEventElapsedTime(out,boundary_events[0],boundary_events[boundary_markers.size()]);if(code)return code;
  for(size_t j=0;j<boundary_markers.size();++j){
    code=synEventElapsedTime(out+3+j,boundary_events[j],boundary_events[j+1]);if(code)return code;
  }
  return 0;
}
