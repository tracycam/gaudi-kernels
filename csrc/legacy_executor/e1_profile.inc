// Direct Synapse trace. Export strings while the API's pointer-bearing buffer
// is alive; the resulting JSONL is portable. No PyTorch profiler callbacks.
static bool profile_started=false;
static void profile_string(FILE* out,const char* s){
  fputc('"',out);
  if(s)for(const unsigned char* p=(const unsigned char*)s;*p;++p){
    if(*p=='"'||*p=='\\'){fputc('\\',out);fputc(*p,out);}
    else if(*p<32)fprintf(out,"\\u%04x",*p);
    else fputc(*p,out);
  }
  fputc('"',out);
}
extern "C" int e1_profile_required(uint32_t* bytes){
  Guard guard;return synProfilerQueryRequiredMemory(device,bytes);
}
extern "C" int e1_profile_start(uint64_t address,uint64_t bytes){
  if(profile_started)return -1;
  Guard guard;uint32_t required=0;
  int code=synProfilerQueryRequiredMemory(device,&required);if(code)return code;
  if(required){
    if(!address||bytes<required)return -2;
    code=synProfilerSetUserBuffer(device,(void*)address);if(code)return code;
  }
  code=synProfilerStart(synTraceAll,device);
  profile_started=code==0;return code;
}
extern "C" int e1_profile_stop(const char* path,uint64_t* counts){
  if(!profile_started)return -1;
  Guard guard;
  int code=synProfilerStop(synTraceAll,device);profile_started=false;if(code)return code;
  size_t bytes=0,count=0;
  code=synProfilerGetTrace(synTraceAll,device,synTraceFormatTEF,nullptr,&bytes,&count);
  if(code)return code;
  counts[0]=bytes;counts[1]=count;
  if(!bytes||count<2)return -2;
  std::vector<unsigned char> buffer(bytes);
  code=synProfilerGetTrace(synTraceAll,device,synTraceFormatTEF,buffer.data(),&bytes,&count);
  if(code)return code;
  if(count*sizeof(synTraceEvent)>bytes)return -3;
  FILE* out=fopen(path,"w");if(!out)return -4;
  const auto* events=(const synTraceEvent*)buffer.data();
  // The installed bridge parser also excludes the final sentinel entry.
  for(size_t i=0;i+1<count;++i){const auto& e=events[i];
    fprintf(out,"{\"name\":");profile_string(out,e.name);
    fprintf(out,",\"cat\":");profile_string(out,e.category);
    char phase[2]={e.type,0};fprintf(out,",\"ph\":");profile_string(out,phase);
    fprintf(out,",\"ts\":%.9Lf,\"dur\":%.9Lf,\"pid\":%u,\"tid\":%u,\"id\":%llu,\"args\":{",
      e.timestamp,e.duration,e.engineType,e.engineIndex,(unsigned long long)e.contextId);
    if(e.type=='M'){
      fprintf(out,"\"name\":");profile_string(out,e.arguments.name);
      fprintf(out,",\"group\":");profile_string(out,e.arguments.group);
    }else if(e.type=='X'||e.type=='B'||e.type=='E'){
      fprintf(out,"\"recipe_id\":%u,\"recipe_name\":",e.arguments.recipeId);
      profile_string(out,e.arguments.recipeName);
      fprintf(out,",\"op\":");profile_string(out,e.arguments.operation);
      fprintf(out,",\"dtype\":");profile_string(out,e.arguments.dataType);
      fprintf(out,",\"stream\":%llu,\"event\":%llu",(unsigned long long)e.arguments.streamHandle,(unsigned long long)e.arguments.eventHandle);
    }
    fprintf(out,"}}\n");
  }
  const bool failed=ferror(out);fclose(out);return failed?-5:0;
}
