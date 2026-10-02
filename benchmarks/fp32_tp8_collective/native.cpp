// Owned allocations/recipe/communicator. Never borrow PyTorch physical bindings.
#include <synapse_api.h>
#include <hccl.h>
#include "../../csrc/common/experiment_device.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
using Clock=std::chrono::steady_clock;
static constexpr int N=6144,Stages=69,World=8;
static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"original IEEE FP32 required");
static constexpr uint64_t Bytes=N*4,MatrixBytes=Bytes*Stages;
static long long ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static void ck(synStatus s,const char*w){if(s!=synSuccess){std::fprintf(stderr,"Synapse %s status%d\n",w,int(s));std::exit(2);}}
static void hc(hcclResult_t s,const char*w){if(s!=hcclSuccess){std::fprintf(stderr,"HCCL %s status%d\n",w,int(s));std::exit(3);}}
struct Shared{std::atomic<int>ready,arrived,phase;hcclUniqueId uid;};
static_assert(std::atomic<int>::is_always_lock_free,"Linux shared memory barrier requires lock-free atomics");
struct SumRecipe{
 synGraphHandle graph{};synRecipeHandle recipe{};synTensor tensors[2]{};synSectionHandle sections[2]{};synLaunchTensorInfo bindings[2]{};uint64_t workspace=0;
 SumRecipe(synDeviceId dev,int ranks,const std::string&name,bool gate=false){
  ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");const char*names[]={"gathered","sum"};
  for(int i=0;i<2;++i){ck(synSectionCreate(&sections[i],0,graph),"section");ck(synSectionSetPersistent(sections[i],true),"persistent");synTensorDescriptor d{};d.m_name=names[i];d.m_dataType=syn_type_single;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=gate?64:N;d.m_sizes[1]=d.m_minSizes[1]=(i||gate)?1:ranks;ck(synTensorCreate(&tensors[i],&d,sections[i],0),"tensor");bindings[i].tensorName=names[i];bindings[i].tensorType=DATA_TENSOR;}
  ck(synNodeCreate(graph,tensors,tensors+1,1,1,&ranks,4,gate?"gk_probe_tp8_gate_f32":"gk_probe_tp8_sum_f32",gate?"prequeue_gate":"ordered_fp32_sum",nullptr,nullptr),"node");ck(synGraphCompile(&recipe,graph,name.c_str(),nullptr),"compile");uint64_t bytes=0;ck(synWorkspaceGetSize(&bytes,recipe),"workspace size");if(bytes)ck(synDeviceMalloc(dev,bytes,0,0,&workspace),"workspace allocation");uint64_t ids[2];ck(synTensorRetrieveIds(recipe,names,ids,2),"ids");for(int i=0;i<2;++i)bindings[i].tensorId=ids[i];
 }
 void launch(synStreamHandle stream,uint64_t input,uint64_t output){bindings[0].pTensorAddress=input;bindings[1].pTensorAddress=output;ck(synLaunch(stream,bindings,2,workspace,recipe,0),"sum launch");}
 void close(synDeviceId dev){ck(synRecipeDestroy(recipe),"recipe destroy");for(int i=0;i<2;++i){ck(synTensorDestroy(tensors[i]),"tensor destroy");ck(synSectionDestroy(sections[i]),"section destroy");}ck(synGraphDestroy(graph),"graph destroy");if(workspace)ck(synDeviceFree(dev,workspace,0),"workspace free");}
};
template<class T>std::vector<T> read(const std::string&path,size_t count){std::vector<T>v(count);std::ifstream f(path,std::ios::binary);f.read((char*)v.data(),count*sizeof(T));if(!f||f.peek()!=EOF){std::fprintf(stderr,"bad fixture %s\n",path.c_str());std::exit(4);}return v;}
int main(int argc,char**argv){
 if(argc!=6){std::fprintf(stderr,"native OUT FIXTURE RANK REPEATS TRIALS\n");return 1;}
 std::string out=argv[1],fixture=argv[2];int rank=std::atoi(argv[3]),repeats=std::atoi(argv[4]),trials=std::atoi(argv[5]);if(rank<0||rank>=World||repeats<1||repeats>100||trials<1||trials>20)return 1;
 const char*comparison=std::getenv("TP8_PROBE_COMPARISON");
 bool prequeue=std::getenv("TP8_PROBE_PREQUEUE")&&std::strcmp(std::getenv("TP8_PROBE_PREQUEUE"),"1")==0;
 bool handoff=comparison&&std::strcmp(comparison,"ar_handoff")==0;
 bool paired_consumers=comparison&&std::strcmp(comparison,"ar_ag_consumers")==0;
 bool inplace_comparison=comparison&&std::strcmp(comparison,"ag_inplace")==0;
 bool two_stream=comparison&&std::strcmp(comparison,"ag_two_stream")==0;
 bool packet_only=std::getenv("TP8_PROBE_PACKET_ONLY")&&std::strcmp(std::getenv("TP8_PROBE_PACKET_ONLY"),"1")==0;
 if(comparison&&!handoff&&!paired_consumers&&!inplace_comparison&&!two_stream&&std::strcmp(comparison,"ag_sum"))return 1;
 if(packet_only&&(!(inplace_comparison||two_stream)||prequeue))return 1;
 if(two_stream&&repeats>4)return 1;
 const int active_stages=packet_only?1:Stages;
 int validation_failed=0;
 if(const char* pin=std::getenv("TP8_PROBE_CPU")){int cpu=std::atoi(pin);if(cpu<0||cpu>=CPU_SETSIZE)return 1;cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);if(sched_setaffinity(0,sizeof(mask),&mask))return 1;}
 auto lifecycle=[&](const char*stage){auto path=out+"/rank"+std::to_string(rank)+"-lifecycle.json";auto temp=path+".tmp";std::ofstream f(temp);f<<"{\"stage\":\""<<stage<<"\",\"monotonic_ns\":"<<ns()<<"}\n";f.close();if(!f||std::rename(temp.c_str(),path.c_str()))std::exit(9);};
 lifecycle("fixture_load");
 int fd=open((out+"/shared").c_str(),O_RDWR);if(fd<0)return 4;auto*sh=(Shared*)mmap(nullptr,65536,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);if(sh==MAP_FAILED)return 4;
 auto wait=[&](auto predicate){auto deadline=Clock::now()+std::chrono::seconds(30);while(!predicate()){if(Clock::now()>deadline){std::fprintf(stderr,"bounded peer wait expired\n");std::exit(5);}std::this_thread::yield();}};
 auto barrier=[&](){int phase=sh->phase.load(std::memory_order_acquire);if(sh->arrived.fetch_add(1,std::memory_order_acq_rel)==World-1){sh->arrived.store(0);sh->phase.fetch_add(1,std::memory_order_release);}else wait([&](){return sh->phase.load(std::memory_order_acquire)!=phase;});};
 auto input=read<float>(fixture+"/input-rank"+std::to_string(rank)+".bin",Stages*N);
 auto audit_begin=(int(*)(const char*))dlsym(RTLD_DEFAULT,"tp8_audit_begin");auto audit_end=(int(*)())dlsym(RTLD_DEFAULT,"tp8_audit_end");if(!audit_begin||!audit_end){std::fprintf(stderr,"API audit preload required\n");return 4;}
 auto ref=read<double>(fixture+"/oracle-fp64.bin",Stages*N),absolute=read<double>(fixture+"/absolute-fp64.bin",Stages*N);auto ordered=read<float>(fixture+"/ordered-fp32.bin",Stages*N);auto all=read<float>(fixture+"/all-inputs.bin",World*Stages*N);
 unsigned module=gaudi_experiment_module();lifecycle("initialize");ck(synInitialize(),"initialize");synDeviceId dev;lifecycle("acquire");ck(synDeviceAcquireByModuleId(&dev,module),"acquire explicit module");
 if(rank==0){hc(hcclGetUniqueId(&sh->uid),"unique id");sh->ready.store(1,std::memory_order_release);}else wait([&](){return sh->ready.load(std::memory_order_acquire)!=0;});
 hcclComm_t comm;hc(hcclCommInitRank(&comm,World,sh->uid,rank),"init rank");synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"begin");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"end");
 uint64_t a,b,g;void*host;ck(synHostMalloc(dev,MatrixBytes+Bytes*World,0,&host),"host");ck(synDeviceMalloc(dev,MatrixBytes,0,0,&a),"input");ck(synDeviceMalloc(dev,MatrixBytes,0,0,&b),"output");ck(synDeviceMalloc(dev,Bytes*(World+6),0,0,&g),"gather scratch");std::memcpy(host,input.data(),MatrixBytes);ck(synMemCopyAsync(stream,(uint64_t)host,MatrixBytes,a,HOST_TO_DRAM),"upload");ck(synStreamSynchronize(stream),"upload sync");
 lifecycle("compile_and_execute");
 synEventHandle gate_begin{};SumRecipe*gate=nullptr;SumRecipe*marker=nullptr;SumRecipe*producer=nullptr;
 if(paired_consumers||inplace_comparison||two_stream)producer=new SumRecipe(dev,1,out+"/producer-rank"+std::to_string(rank));
 if(prequeue){marker=new SumRecipe(dev,1,out+"/terminal-marker-rank"+std::to_string(rank));gate=new SumRecipe(dev,50000000,out+"/gate-rank"+std::to_string(rank),true);ck(synEventCreate(&gate_begin,dev,EVENT_COLLECT_TIME),"gate event");ck(synMemCopyAsync(stream,a,256,g+10*Bytes,DRAM_TO_DRAM),"gate input");ck(synStreamSynchronize(stream),"gate input sync");}
 // Each outstanding stage owns distinct events until the enclosing enqueue
 // finishes. No event is re-recorded while a consumer may still reference it.
 synStreamHandle collective_stream{};
 std::vector<synEventHandle> produced_events, gathered_events;
 if(two_stream){
  ck(synStreamCreateGeneric(&collective_stream,dev,0),"collective stream");
  size_t count=size_t(active_stages)*std::max(repeats,5);
  produced_events.resize(count);gathered_events.resize(count);
  for(size_t i=0;i<count;++i){
   ck(synEventCreate(&produced_events[i],dev,0),"producer event");
   ck(synEventCreate(&gathered_events[i],dev,0),"gather event");
  }
 }
 SumRecipe sum(dev,handoff?1:8,out+"/sum-rank"+std::to_string(rank));std::ofstream log(out+"/rank"+std::to_string(rank)+".jsonl");
 // The handoff diagnostic uses two disjoint scratch rows and the same stream:
 // original input -> TPC(+0) -> AR -> TPC(+0) -> preserved output. The one-rank
 // sum is FP32 addition with zero, not a claimed bitwise identity for -0 or
 // subnormals. Exact/ordinary fixtures remain subject to their existing gates.
 auto enqueue=[&](bool alternative,int count){for(int replay=0;replay<count;++replay)for(int stage=0;stage<active_stages;++stage){
  auto src=(void*)(a+stage*Bytes),dst=(void*)(b+stage*Bytes);
  if(two_stream){
   producer->launch(stream,(uint64_t)src,g+World*Bytes);
   if(alternative){
    size_t index=size_t(replay)*active_stages+stage;
    ck(synEventRecord(produced_events[index],stream),"record producer done");
    ck(synStreamWaitEvent(collective_stream,produced_events[index],0),"AG waits producer");
    hc(hcclAllGather((void*)(g+World*Bytes),(void*)g,N,hcclFloat,comm,collective_stream),"explicit stream AG");
    ck(synEventRecord(gathered_events[index],collective_stream),"record AG done");
    ck(synStreamWaitEvent(stream,gathered_events[index],0),"sum waits AG");
   }else{
    hc(hcclAllGather((void*)(g+World*Bytes),(void*)g,N,hcclFloat,comm,stream),"serial stream AG");
   }
   // This also orders scratch reuse by the next producer/AG.
   sum.launch(stream,g,(uint64_t)dst);
  }else if(inplace_comparison){
   // Exactly the same producer/AG/consumer. Only the owned producer output
   // binding changes: rank-slot output is HCCL's public in-place AG contract.
   uint64_t send=g+(alternative?rank:World)*Bytes;
   producer->launch(stream,(uint64_t)src,send);
   hc(hcclAllGather((void*)send,(void*)g,N,hcclFloat,comm,stream),"inplace comparison AG FP32/6144");
   sum.launch(stream,g,(uint64_t)dst);
  }else if(paired_consumers){producer->launch(stream,(uint64_t)src,g+8*Bytes);if(alternative){hc(hcclAllGather((void*)(g+8*Bytes),(void*)g,N,hcclFloat,comm,stream),"paired allgather FP32/6144");sum.launch(stream,g,(uint64_t)dst);}else{hc(hcclAllReduce((void*)(g+8*Bytes),(void*)(g+9*Bytes),N,hcclFloat,hcclSum,comm,stream),"paired allreduce FP32/6144");producer->launch(stream,g+9*Bytes,(uint64_t)dst);}}else if(alternative&&handoff){sum.launch(stream,(uint64_t)src,g);hc(hcclAllReduce((void*)g,(void*)(g+Bytes),N,hcclFloat,hcclSum,comm,stream),"interleaved allreduce FP32/6144");sum.launch(stream,g+Bytes,(uint64_t)dst);}else if(alternative){hc(hcclAllGather(src,(void*)g,N,hcclFloat,comm,stream),"allgather FP32/6144");sum.launch(stream,g,(uint64_t)dst);}else hc(hcclAllReduce(src,dst,N,hcclFloat,hcclSum,comm,stream),"allreduce FP32/6144");}};
 auto verify=[&](bool gather,const std::string&tag){ck(synMemCopyAsync(stream,b,MatrixBytes,(uint64_t)host,DRAM_TO_HOST),"download outputs");ck(synStreamSynchronize(stream),"download sync");auto*y=(float*)host;size_t bad=0,ordered_bad=0,unexpected_zero=0;double max_abs=0,max_back=0;constexpr double eps=std::numeric_limits<float>::epsilon(),gamma=7*eps/(1-7*eps);
  for(size_t i=0;i<size_t(active_stages*N);++i){double delta=std::abs(double(y[i])-ref[i]);max_abs=std::max(max_abs,delta);max_back=std::max(max_back,delta/std::max(absolute[i],std::numeric_limits<double>::min()));bad+=!std::isfinite(y[i])||delta>gamma*absolute[i];ordered_bad+=std::memcmp(y+i,ordered.data()+i,4)!=0;unexpected_zero+=ref[i]!=0&&y[i]==0;}
  std::ofstream raw(out+"/rank"+std::to_string(rank)+"-"+tag+".bin",std::ios::binary);raw.write((char*)host,active_stages*Bytes);raw.close();
  if(gather){ck(synMemCopyAsync(stream,g,Bytes*World,(uint64_t)host,DRAM_TO_HOST),"gather byte check");ck(synStreamSynchronize(stream),"gather check sync");for(int r=0;r<World;++r)bad+=std::memcmp((char*)host+r*Bytes,all.data()+(r*Stages+active_stages-1)*N,Bytes)!=0;bad+=ordered_bad;std::ofstream gathered(out+"/rank"+std::to_string(rank)+"-"+tag+"-gathered.bin",std::ios::binary);gathered.write((char*)host,Bytes*World);}
  ck(synMemCopyAsync(stream,a,MatrixBytes,(uint64_t)host,DRAM_TO_HOST),"input immutability");ck(synStreamSynchronize(stream),"input sync");bad+=std::memcmp(host,input.data(),MatrixBytes)!=0;
  log<<"{\"stage\":\"quality\",\"tag\":\""<<tag<<"\",\"rank\":"<<rank<<",\"bad\":"<<bad<<",\"checked\":"<<active_stages*N<<",\"max_abs\":"<<max_abs<<",\"max_backward_error\":"<<max_back<<",\"ordered_FP32_bit_differences\":"<<ordered_bad<<",\"unexpected_zero\":"<<unexpected_zero<<",\"gather_bytes_exact\":"<<(gather&&bad==0?"true":"false")<<"}\n";log.flush();if(bad)validation_failed=6;
 };
 for(int phase=0;phase<4;++phase){bool gather=phase==1||phase==2;std::string tag=(two_stream?(gather?"agdual":"agserial"):inplace_comparison?(gather?"agin":"agout"):paired_consumers?(gather?"agconsumer":"arconsumer"):(gather?(handoff?"handoff":"ags"):"ar"))+std::to_string(phase);barrier();if(inplace_comparison||two_stream){ck(synMemsetD32Async(g,0x7fc12345u,World*N,stream),"poison gather scratch");ck(synStreamSynchronize(stream),"poison sync");}if(audit_begin((out+"/rank"+std::to_string(rank)+"-"+tag+"-api.jsonl").c_str()))return 8;enqueue(gather,1);ck(synStreamSynchronize(stream),"quality sync");if(audit_end())return 8;verify(inplace_comparison||two_stream||(gather&&!handoff),tag);barrier();enqueue(gather,packet_only?0:5);ck(synStreamSynchronize(stream),"warmup sync");
  for(int trial=0;trial<(packet_only?0:trials);++trial){barrier();int cpu_before=sched_getcpu();auto start=ns();if(prequeue){ck(synEventRecord(gate_begin,stream),"gate begin record");gate->launch(stream,g+10*Bytes,g+11*Bytes);hc(hcclAllReduce((void*)(g+11*Bytes),(void*)(g+12*Bytes),1,hcclFloat,hcclSum,comm,stream),"gate rank alignment");}ck(synEventRecord(begin,stream),"begin record");enqueue(gather,repeats);if(marker)marker->launch(stream,b+(Stages-1)*Bytes,g+13*Bytes);auto submitted=ns();ck(synEventRecord(end,stream),"end record");int gate_query=prequeue?int(synEventQuery(begin)):-1;ck(synEventSynchronize(end),"event sync");ck(synEventSynchronize(begin),"begin event sync");ck(synStreamSynchronize(stream),"final sync");auto finish=ns();uint64_t elapsed=0;ck(synEventElapsedTime(&elapsed,begin,end),"elapsed");uint64_t gate_ns=0;if(prequeue)ck(synEventElapsedTime(&gate_ns,gate_begin,begin),"gate elapsed");double event=elapsed/1000.,wall=(finish-start)/1000.;bool valid=event>0&&event<=wall*1.1+.25&&(prequeue?gate_query==int(synBusy):wall<=std::max(event*1.35,event+2));
   log<<"{\"stage\":\"timing\",\"rank\":"<<rank<<",\"module\":"<<module<<",\"start_ns\":"<<start<<",\"submit_ns\":"<<submitted<<",\"finish_ns\":"<<finish<<",\"cpu_before\":"<<cpu_before<<",\"cpu_after\":"<<sched_getcpu()<<",\"phase\":"<<phase<<",\"mode\":\""<<(two_stream?(gather?"TPC_FP32_AG_TWO_STREAM_sum":"TPC_FP32_AG_SERIAL_sum"):inplace_comparison?(gather?"TPC_FP32_AG_INPLACE_sum":"TPC_FP32_AG_OUTPLACE_sum"):paired_consumers?(gather?"TPC_FP32_AG_sum":"TPC_FP32_AR_copy"):(gather?(handoff?"TPC_FP32_AR_TPC":"FP32_AG_sum"):"FP32_AR"))<<"\",\"trial\":"<<trial<<",\"repeats\":"<<repeats<<",\"collectives\":"<<Stages*repeats<<",\"compute_recipe_launches\":"<<((paired_consumers||inplace_comparison||two_stream)?2*Stages*repeats:(gather?Stages*repeats*(handoff?2:1):0))<<",\"hccl_dtype\":"<<int(hcclFloat)<<",\"hccl_count\":"<<N<<",\"send_bytes\":"<<Bytes<<",\"event_us\":"<<event<<",\"wall_us\":"<<wall<<",\"submit_us\":"<<(submitted-start)/1000.<<",\"terminal_marker_launches\":"<<(prequeue?1:0)<<",\"gate_us\":"<<gate_ns/1000.<<",\"gate_query\":"<<gate_query<<",\"prequeued\":"<<(prequeue?"true":"false")<<",\"timing_valid\":"<<(valid?"true":"false")<<"}\n";log.flush();
   // A rejected timing sample is not a device execution failure. Keep it and
   // finish collective order and ordinary teardown on every rank. Exiting
   // here abandons live HCL threads; summarize.py still rejects any !valid.
  }verify(inplace_comparison||two_stream||(gather&&!handoff),tag+"-post");barrier();
 }
 lifecycle("free_owned_resources");
 if(two_stream){
  ck(synStreamSynchronize(collective_stream),"collective final sync");
  for(auto event:produced_events)ck(synEventDestroy(event),"producer event destroy");
  for(auto event:gathered_events)ck(synEventDestroy(event),"gather event destroy");
  ck(synStreamDestroy(collective_stream),"collective stream destroy");
 }
 sum.close(dev);if(producer){producer->close(dev);delete producer;}if(marker){marker->close(dev);delete marker;}if(gate){gate->close(dev);delete gate;ck(synEventDestroy(gate_begin),"gate event destroy");}ck(synDeviceFree(dev,a,0),"free a");ck(synDeviceFree(dev,b,0),"free b");ck(synDeviceFree(dev,g,0),"free g");ck(synHostFree(dev,host,0),"host free");hc(hcclCommDestroy(comm),"comm destroy");ck(synEventDestroy(begin),"begin destroy");ck(synEventDestroy(end),"end destroy");ck(synStreamDestroy(stream),"stream destroy");lifecycle("device_release");ck(synDeviceRelease(dev),"device release");lifecycle("synapse_destroy");ck(synDestroy(),"destroy");lifecycle("complete");munmap(sh,65536);close(fd);return validation_failed;
}
