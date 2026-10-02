// SPDX-License-Identifier: Apache-2.0
// Own Synapse compute graph; no PyTorch, recipe interposition or borrowed tensors.
#include <synapse_api.h>
#include "../../csrc/common/experiment_device.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>
static constexpr int N=6144;static constexpr uint64_t Bytes=N*4;
static void ck(synStatus s,const char*w){if(s!=synSuccess){std::fprintf(stderr,"%s: %d\n",w,int(s));std::exit(2);}}
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Recipe{
 synGraphHandle graph{};synRecipeHandle recipe{};std::vector<synTensor>tensors;std::vector<synSectionHandle>sections;std::vector<std::string>names;uint64_t workspace=0; synLaunchTensorInfo bindings[2]{};
 Recipe(synDeviceId dev,int nodes,const std::string&name){
  ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");tensors.resize(nodes+1);sections.resize(nodes+1);names.reserve(nodes+1);
  for(int i=0;i<=nodes;++i){names.push_back("value"+std::to_string(i));bool persistent=i==0||i==nodes;
   if(persistent){ck(synSectionCreate(&sections[i],0,graph),"section");ck(synSectionSetPersistent(sections[i],true),"persistent");}
   synTensorDescriptor d{};d.m_name=names.back().c_str();d.m_dataType=syn_type_single;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=N;d.m_sizes[1]=d.m_minSizes[1]=1;
   ck(synTensorCreate(&tensors[i],&d,sections[i],0),"tensor");
  }
  float increment=.03125f;
  for(int i=0;i<nodes;++i)ck(synNodeCreate(graph,&tensors[i],&tensors[i+1],1,1,&increment,4,"gk_native_graph_affine_f32_v1",("add"+std::to_string(i)).c_str(),nullptr,nullptr),"node");
  ck(synGraphCompile(&recipe,graph,name.c_str(),nullptr),"compile");uint64_t bytes=0;ck(synWorkspaceGetSize(&bytes,recipe),"workspace size");if(bytes)ck(synDeviceMalloc(dev,bytes,0,0,&workspace),"workspace alloc");
  const char*external[]={names.front().c_str(),names.back().c_str()};uint64_t ids[2];ck(synTensorRetrieveIds(recipe,external,ids,2),"ids");
  for(int i=0;i<2;++i){bindings[i].tensorName=external[i];bindings[i].tensorId=ids[i];bindings[i].tensorType=DATA_TENSOR;}
 }
 void launch(synStreamHandle stream,uint64_t input,uint64_t output){bindings[0].pTensorAddress=input;bindings[1].pTensorAddress=output;ck(synLaunch(stream,bindings,2,workspace,recipe,0),"launch");}
 void close(synDeviceId dev){ck(synRecipeDestroy(recipe),"recipe free");for(auto t:tensors)ck(synTensorDestroy(t),"tensor free");for(auto s:sections)if(s)ck(synSectionDestroy(s),"section free");ck(synGraphDestroy(graph),"graph free");if(workspace)ck(synDeviceFree(dev,workspace,0),"workspace free");}
};
int main(int argc,char**argv){
 if(argc!=2)return 1;
 std::string out=argv[1];std::ofstream log(out+"/records.jsonl");
 using Begin=int(*)(int,const char*);using End=int(*)(uint64_t*);auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"ng_scope_begin"));auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"ng_scope_end"));if(!begin||!end)return 1;
 ck(synInitialize(),"initialize");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"acquire");synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");
 uint64_t a,b,c;void*host;ck(synDeviceMalloc(dev,Bytes,0,0,&a),"a");ck(synDeviceMalloc(dev,Bytes,0,0,&b),"b");ck(synDeviceMalloc(dev,Bytes,0,0,&c),"c");ck(synHostMalloc(dev,Bytes*2,0,&host),"host");auto*x=static_cast<float*>(host);auto*y=x+N;
 Recipe single(dev,1,out+"/one");synEventHandle ev0,ev1;ck(synEventCreate(&ev0,dev,EVENT_COLLECT_TIME),"event0");ck(synEventCreate(&ev1,dev,EVENT_COLLECT_TIME),"event1");
 for(int nodes:{2,8,32,64}){
  Recipe combined(dev,nodes,out+"/combined"+std::to_string(nodes));
  int phase=0;
  for(int variant:{0,1,2,3,3,2,1,0}){
   auto enqueue=[&](){if(variant==1)combined.launch(stream,a,c);else{for(int i=0;i<nodes;++i)single.launch(stream,i?(i%2?b:c):a,i%2?c:b);}};
   for(int mutation=0;mutation<4;++mutation){
    for(int i=0;i<N;++i)x[i]=float((i*17+mutation*7)%131-65)/32;
    ck(synMemCopyAsync(stream,(uint64_t)x,Bytes,a,HOST_TO_DRAM),"input");ck(synMemsetD32Async(b,0x7fc12345u,N,stream),"poison b");ck(synMemsetD32Async(c,0x7fc12345u,N,stream),"poison c");ck(synStreamSynchronize(stream),"input sync");
    std::string file=out+"/trace-n"+std::to_string(nodes)+"-v"+std::to_string(variant)+"-p"+std::to_string(phase)+"-m"+std::to_string(mutation)+".jsonl";
    if(begin(variant>=2?variant-1:0,file.c_str()))return 3;
    enqueue();uint64_t stats[5];if(end(stats))return 4;ck(synStreamSynchronize(stream),"gate sync");
    ck(synMemCopyAsync(stream,c,Bytes,(uint64_t)y,DRAM_TO_HOST),"output");ck(synStreamSynchronize(stream),"output sync");int bad=0;
    for(int i=0;i<N;++i){float expected=x[i]+nodes*.03125f;if(std::memcmp(&expected,&y[i],4))++bad;}
    log<<"{\"stage\":\"gate\",\"nodes\":"<<nodes<<",\"variant\":"<<variant<<",\"phase\":"<<phase<<",\"mutation\":"<<mutation<<",\"bad\":"<<bad<<",\"synapse_launches\":"<<(variant==1?1:nodes)<<",\"scal_requests\":"<<stats[0]<<",\"scal_calls\":"<<stats[1]<<",\"flushes\":"<<stats[2]<<",\"wait_flushes\":"<<stats[3]<<",\"foreign_submits\":"<<stats[4]<<"}\n";log.flush();if(bad)return 5;
   }
   for(int i=0;i<5;++i){uint64_t stats[5];if(variant>=2&&begin(variant-1,nullptr))return 3;enqueue();if(variant>=2&&end(stats))return 4;}ck(synStreamSynchronize(stream),"warm sync");
   for(int trial=0;trial<5;++trial){ck(synEventRecord(ev0,stream),"begin");auto started=ns();uint64_t stats[5]{};
    for(int rep=0;rep<100;++rep){if(variant>=2&&begin(variant-1,nullptr))return 3;enqueue();if(variant>=2&&end(stats))return 4;}
    auto submitted=ns();ck(synEventRecord(ev1,stream),"end");ck(synEventSynchronize(ev1),"event sync");ck(synStreamSynchronize(stream),"timing sync");auto finished=ns();uint64_t device=0;ck(synEventElapsedTime(&device,ev0,ev1),"elapsed");
    log<<"{\"stage\":\"timing\",\"nodes\":"<<nodes<<",\"variant\":"<<variant<<",\"phase\":"<<phase<<",\"trial\":"<<trial<<",\"repeats\":100,\"event_us\":"<<device/100000.<<",\"wall_us\":"<<(finished-started)/100000.<<",\"submit_us\":"<<(submitted-started)/100000.<<"}\n";log.flush();
   }
   ++phase;
  }combined.close(dev);
 }
 single.close(dev);ck(synEventDestroy(ev0),"event free");ck(synEventDestroy(ev1),"event free");ck(synStreamDestroy(stream),"stream free");for(auto p:{a,b,c})ck(synDeviceFree(dev,p,0),"device free");ck(synHostFree(dev,host,0),"host free");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;
}
