#include "probe_helpers.hpp"
#include <hccl.h>
#include <atomic>
#include <cmath>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
constexpr int World=8,Stages=69;constexpr uint64_t MatrixBytes=Bytes*Stages,GatherBytes=Bytes*World;
void hc(hcclResult_t s,const char*w){if(s!=hcclSuccess){std::cerr<<w<<": "<<s<<'\n';std::exit(3);}}
struct Shared{std::atomic<int>ready,arrived,phase;hcclUniqueId uid;};
static_assert(std::atomic<int>::is_always_lock_free,"shared barrier requires lock-free atomics");
float value(bool ordinary,int rank,int stage,int i,int mutation){
 if(!ordinary)return float((i*17+rank*7+stage*11+mutation*19)%131-65)/32.f;
 return std::sin(float(i+rank*17+stage*3+mutation*29))*.15f+float(i%13)*.01337f;
}
struct Sum {
 synGraphHandle graph{};synRecipeHandle recipe{};synTensor tensors[2]{};synSectionHandle sections[2]{};synLaunchTensorInfo bindings[2]{};uint64_t workspace=0;
 Sum(synDeviceId dev,const std::string&path){ck(synGraphCreate(&graph,synDeviceGaudi2),"sum graph");const char*names[]={"gathered","sum"};
  for(int i=0;i<2;++i){ck(synSectionCreate(&sections[i],0,graph),"sum section");ck(synSectionSetPersistent(sections[i],true),"sum persistent");synTensorDescriptor d{};d.m_name=names[i];d.m_dataType=syn_type_single;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=N;d.m_sizes[1]=d.m_minSizes[1]=i?1:World;ck(synTensorCreate(&tensors[i],&d,sections[i],0),"sum tensor");bindings[i].tensorName=names[i];bindings[i].tensorType=DATA_TENSOR;}
  int ranks=World;ck(synNodeCreate(graph,tensors,tensors+1,1,1,&ranks,4,"gk_probe_tp8_sum_f32","ordered_sum",nullptr,nullptr),"sum node");ck(synGraphCompile(&recipe,graph,path.c_str(),nullptr),"sum compile");uint64_t size;ck(synWorkspaceGetSize(&size,recipe),"sum workspace");if(size)ck(synDeviceMalloc(dev,size,0,0,&workspace),"sum workspace allocation");uint64_t ids[2];ck(synTensorRetrieveIds(recipe,names,ids,2),"sum ids");for(int i=0;i<2;++i)bindings[i].tensorId=ids[i];
 }
 void launch(synStreamHandle stream,uint64_t input,uint64_t output){bindings[0].pTensorAddress=input;bindings[1].pTensorAddress=output;ck(synLaunch(stream,bindings,2,workspace,recipe,0),"sum launch");}
 void close(synDeviceId dev){ck(synRecipeDestroy(recipe),"sum recipe free");for(auto t:tensors)ck(synTensorDestroy(t),"sum tensor free");for(auto s:sections)ck(synSectionDestroy(s),"sum section free");ck(synGraphDestroy(graph),"sum graph free");if(workspace)ck(synDeviceFree(dev,workspace,0),"sum workspace free");}
};
int main(int argc,char**argv){
 if(argc!=6)return 1;
 std::string root=argv[1];bool ordinary=std::filesystem::path(argv[2]).filename()=="ordinary";int rank=std::atoi(argv[3]),repeats=std::atoi(argv[4]),trials=std::atoi(argv[5]);if(rank<0||rank>=8||repeats<1||repeats>16)return 1;
 std::ofstream log(root+"/rank"+std::to_string(rank)+".jsonl");auto lifecycle=[&](const char*stage){std::ofstream f(root+"/rank"+std::to_string(rank)+"-lifecycle.json");f<<"{\"stage\":\""<<stage<<"\"}\n";};
 int fd=open((root+"/shared").c_str(),O_RDWR);if(fd<0)return 2;auto*sh=(Shared*)mmap(nullptr,65536,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);if(sh==MAP_FAILED)return 2;
 auto wait=[&](auto predicate){auto deadline=ns()+30000000000ull;while(!predicate()){if(ns()>deadline){std::cerr<<"peer wait timeout\n";std::exit(5);}std::this_thread::yield();}};
 auto barrier=[&](){int phase=sh->phase.load(std::memory_order_acquire);if(sh->arrived.fetch_add(1,std::memory_order_acq_rel)==World-1){sh->arrived=0;sh->phase.fetch_add(1,std::memory_order_release);}else wait([&]{return sh->phase.load(std::memory_order_acquire)!=phase;});};
 lifecycle("initialize");ck(synInitialize(),"initialize");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"acquire");if(rank==0){hc(hcclGetUniqueId(&sh->uid),"uid");sh->ready.store(1,std::memory_order_release);}else wait([&]{return sh->ready.load(std::memory_order_acquire)!=0;});hcclComm_t comm;hc(hcclCommInitRank(&comm,8,sh->uid,rank),"comm init");synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");
 uint64_t a,b,gathered,out;ck(synDeviceMalloc(dev,MatrixBytes,0,0,&a),"input");ck(synDeviceMalloc(dev,Bytes,0,0,&b),"producer");ck(synDeviceMalloc(dev,GatherBytes*Stages,0,0,&gathered),"gather");ck(synDeviceMalloc(dev,MatrixBytes,0,0,&out),"output");void*host;ck(synHostMalloc(dev,GatherBytes*Stages,0,&host),"host");
 Producer producer(1,root+"/producer"+std::to_string(rank));synLaunchTensorInfo binding[2]{};const char*names[]={"v0","v1"};uint64_t ids[2];ck(synTensorRetrieveIds(producer.recipe,names,ids,2),"producer ids");for(int i=0;i<2;++i){binding[i].tensorName=names[i];binding[i].tensorId=ids[i];binding[i].tensorType=DATA_TENSOR;}uint64_t pws=0,pws_bytes;ck(synWorkspaceGetSize(&pws_bytes,producer.recipe),"producer workspace");if(pws_bytes)ck(synDeviceMalloc(dev,pws_bytes,0,0,&pws),"producer workspace allocation");Sum sum(dev,root+"/sum"+std::to_string(rank));
 int sentinel1=-13579,sentinel2=-24680;hc(hcclCommSynDevice(comm,&sentinel1),"device query sentinel 1");hc(hcclCommSynDevice(comm,&sentinel2),"device query sentinel 2");log<<"{\"stage\":\"device_query\",\"acquired\":"<<dev<<",\"returned1\":"<<sentinel1<<",\"returned2\":"<<sentinel2<<"}\n";log.flush();
 auto*graph=create(dev,root,"graph"+std::to_string(rank),16);ok(graph,gkg_buffer(graph,"input",a,MatrixBytes,GKG_EXTERNAL_READ),"graph input");ok(graph,gkg_buffer(graph,"producer",0,Bytes,GKG_TEMP),"graph producer");ok(graph,gkg_buffer(graph,"gathered",0,GatherBytes,GKG_TEMP),"graph gathered");ok(graph,gkg_buffer(graph,"output",out,MatrixBytes,GKG_EXTERNAL_RW),"graph output");ok(graph,gkg_recipe(graph,"producer",producer.recipe),"producer clone");ok(graph,gkg_recipe(graph,"sum",sum.recipe),"sum clone");ok(graph,gkg_comm_on_device(graph,"tp",comm,dev),"comm bind");
 for(int stage=0;stage<Stages;++stage){gkg_binding pb[2]={{sizeof(gkg_binding),1,"v0","input",stage*Bytes},{sizeof(gkg_binding),1,"v1","producer",0}};ok(graph,gkg_launch(graph,"producer",pb,2),"producer plan");ok(graph,gkg_collect(graph,GKG_ALL_GATHER,"tp","producer",0,"gathered",0,N,GKG_F32),"AG plan");gkg_binding sb[2]={{sizeof(gkg_binding),1,"gathered","gathered",0},{sizeof(gkg_binding),1,"sum","output",stage*Bytes}};ok(graph,gkg_launch(graph,"sum",sb,2),"sum plan");}
 ok(graph,gkg_instantiate(graph),"instantiate");void*own_stream;ok(graph,gkg_stream(graph,&own_stream),"owned stream");synEventHandle start,end;ck(synEventCreate(&start,dev,EVENT_COLLECT_TIME),"start");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"end");
 std::vector<uint64_t>tickets;uint64_t ag_host_ns=0;
 auto enqueue=[&](int variant,int count){for(int rep=0;rep<count;++rep){if(variant==2){uint64_t ticket;ok(graph,gkg_replay(graph,nullptr,0,&ticket),"graph replay");tickets.push_back(ticket);}else for(int stage=0;stage<Stages;++stage){uint64_t source=a+stage*Bytes;if(variant==1){binding[0].pTensorAddress=source;binding[1].pTensorAddress=b;ck(synLaunch(stream,binding,2,pws,producer.recipe,0),"producer launch");source=b;}auto before=ns();hc(hcclAllGather((void*)source,(void*)(gathered+(variant==0?stage*GatherBytes:0)),N,hcclFloat,comm,stream),"AG");ag_host_ns+=ns()-before;if(variant==1)sum.launch(stream,gathered,out+stage*Bytes);}}};
 auto drain=[&]{for(auto t:tickets){ok(graph,gkg_wait(graph,t),"graph wait");ok(graph,gkg_release(graph,t),"graph release");}tickets.clear();};
 lifecycle("compile_and_execute");int phase=0;
 for(int variant:{0,1,2,2,1,0}){auto timed=variant==2?static_cast<synStreamHandle>(own_stream):stream;
  for(int mutation=0;mutation<3;++mutation){barrier();for(int stage=0;stage<Stages;++stage)for(int i=0;i<N;++i)((float*)host)[stage*N+i]=value(ordinary,rank,stage,i,mutation);ck(synMemCopyAsync(stream,(uint64_t)host,MatrixBytes,a,HOST_TO_DRAM),"mutated input");ck(synMemsetD32Async(out,0x7fc12345u,Stages*N,stream),"poison output");ck(synMemsetD32Async(gathered,0x7fc12345u,Stages*World*N,stream),"poison gather");ck(synStreamSynchronize(stream),"mutation complete");enqueue(variant,1);ck(synStreamSynchronize(timed),"gate complete");drain();ck(synMemCopyAsync(stream,variant?out:gathered,variant?MatrixBytes:GatherBytes*Stages,(uint64_t)host,DRAM_TO_HOST),"readback");ck(synStreamSynchronize(stream),"readback complete");size_t bad=0;
   for(int stage=0;stage<Stages;++stage)for(int i=0;i<N;++i){if(variant){float expected=0;for(int r=0;r<8;++r)expected+=value(ordinary,r,stage,i,mutation)+.03125f;bad+=std::memcmp(&expected,(float*)host+stage*N+i,4)!=0;}else for(int r=0;r<8;++r){float expected=value(ordinary,r,stage,i,mutation);bad+=std::memcmp(&expected,(float*)host+(stage*World+r)*N+i,4)!=0;}}
   log<<"{\"stage\":\"quality\",\"phase\":"<<phase<<",\"variant\":"<<variant<<",\"mutation\":"<<mutation<<",\"bad\":"<<bad<<",\"checked\":"<<uint64_t(Stages)*N*(variant?1:8)<<"}\n";log.flush();if(bad)return 6;
  }
  for(int trial=0;trial<trials;++trial){barrier();ag_host_ns=0;ck(synEventRecord(start,timed),"start record");auto before=ns();enqueue(variant,repeats);auto submitted=ns();ck(synEventRecord(end,timed),"end record");ck(synEventSynchronize(end),"end wait");drain();auto finished=ns();uint64_t elapsed;ck(synEventElapsedTime(&elapsed,start,end),"elapsed");log<<"{\"stage\":\"timing\",\"phase\":"<<phase<<",\"variant\":"<<variant<<",\"trial\":"<<trial<<",\"collectives\":"<<Stages*repeats<<",\"event_us_per_chain\":"<<elapsed/1000./(Stages*repeats)<<",\"wall_us_per_chain\":"<<(finished-before)/1000./(Stages*repeats)<<",\"submit_us_per_chain\":"<<(submitted-before)/1000./(Stages*repeats)<<",\"ag_host_us\":"<<(variant==2?-1.:ag_host_ns/1000./(Stages*repeats))<<"}\n";log.flush();}
  ++phase;
 }
 ok(graph,gkg_report(graph,(root+"/graph-rank"+std::to_string(rank)+".json").c_str()),"report");ok(graph,gkg_destroy(graph),"graph free");producer.close();sum.close(dev);if(pws)ck(synDeviceFree(dev,pws,0),"producer workspace free");for(auto p:{a,b,gathered,out})ck(synDeviceFree(dev,p,0),"device free");ck(synHostFree(dev,host,0),"host free");ck(synEventDestroy(start),"start free");ck(synEventDestroy(end),"end free");ck(synStreamDestroy(stream),"stream free");hc(hcclCommDestroy(comm),"comm free");lifecycle("device_release");ck(synDeviceRelease(dev),"release");lifecycle("synapse_destroy");ck(synDestroy(),"destroy");lifecycle("complete");munmap(sh,65536);close(fd);return 0;
}
