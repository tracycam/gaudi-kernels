// SPDX-License-Identifier: Apache-2.0
// Complete residual+norm(+explicit quant) recipe. Same-stream device events.
#include <synapse_api.h>
#include "../../csrc/common/experiment_device.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static void ck(synStatus s,const char*w){if(s!=synSuccess){std::fprintf(stderr,"FAIL %s: %d\n",w,int(s));std::exit(2);}}
static void file(const std::string&p,void*d,size_t n,bool write){FILE*f=fopen(p.c_str(),write?"wb":"rb");if(!f||(write?fwrite(d,1,n,f):fread(d,1,n,f))!=n){fprintf(stderr,"file %s\n",p.c_str());std::exit(3);}fclose(f);}
struct Buffer{synTensor t;std::string name;uint64_t bytes,address;void*host;bool output;};
int main(int argc,char**argv){
 if(argc!=6)return 2;std::string mode=argv[1],fixture=argv[4],out=argv[5];unsigned m=std::stoul(argv[2]),h=std::stoul(argv[3]);
 if(!m||!h||h>8192||(mode!="bf16"&&mode!="separate"&&mode!="per_row"&&mode!="block128"))return 2;
 bool quant=mode!="bf16",block=mode=="block128";
 ck(synInitialize(),"init");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"module acquire");
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bufs;
 auto tensor=[&](const std::string&name,synDataType type,std::vector<unsigned>shape,uint64_t bytes,bool output,bool persistent=true){
  synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=shape.size();for(unsigned i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];
  synSectionHandle sec=nullptr;if(persistent){ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");}
  synTensor t;ck(synTensorCreate(&t,&d,sec,0),"tensor");
  if(persistent){Buffer b{t,name,bytes,0,nullptr,output};ck(synHostMalloc(dev,bytes,0,&b.host),"host");ck(synDeviceMalloc(dev,bytes,0,0,&b.address),"device");bufs.push_back(b);if(!output)file(fixture+"/"+name+".bin",b.host,bytes,false);}
  return t;
 };
 auto x=tensor("x",syn_type_bf16,{h,m},uint64_t(h)*m*2,false),r=tensor("residual",syn_type_bf16,{h,m},uint64_t(h)*m*2,false),w=tensor("gamma",syn_type_bf16,{h},h*2,false);
 auto rr=tensor("residual_out",syn_type_bf16,{h,m},uint64_t(h)*m*2,true);
 unsigned groups=(h+127)/128;auto y=tensor(quant?"native_a8":"norm",quant?syn_type_fp8_143:syn_type_bf16,block?std::vector<unsigned>{128,m,groups}:std::vector<unsigned>{h,m},uint64_t(m)*(block?groups*128:h)*(quant?1:2),true);
 synTensor scale=nullptr;if(quant)scale=tensor("native_scale",syn_type_single,block?std::vector<unsigned>{1,m,groups}:std::vector<unsigned>{1,m},uint64_t(m)*(block?groups:1)*4,true);
 float eps=1e-6f;synTensor ins[]={x,r,w};
 if(mode=="separate"){
  auto temp=tensor("norm_intermediate",syn_type_bf16,{h,m},0,true,false);synTensor outs[]={rr,temp};ck(synNodeCreate(graph,ins,outs,3,2,&eps,4,"gk_residual_rmsnorm_bf16_v1","norm",nullptr,nullptr),"norm node");synTensor qo[]={y,scale};ck(synNodeCreate(graph,&temp,qo,1,2,nullptr,0,"fp8_linear_activation_fast","quant",nullptr,nullptr),"quant node");
 }else{synTensor outs[]={rr,y,scale};ck(synNodeCreate(graph,ins,outs,3,quant?3:2,&eps,4,block?"gk_residual_rmsnorm_block128_a8_v1":(quant?"gk_residual_rmsnorm_per_row_a8_v1":"gk_residual_rmsnorm_bf16_v1"),"residual_norm",nullptr,nullptr),"fused node");}
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,("norm_"+mode).c_str(),nullptr),"compile");synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");
 std::vector<synLaunchTensorInfo>launch(bufs.size());std::vector<const char*>names;std::vector<uint64_t>ids(bufs.size());
 for(auto&b:bufs)names.push_back(b.name.c_str());ck(synTensorRetrieveIds(recipe,names.data(),ids.data(),ids.size()),"tensor ids");
 for(size_t i=0;i<bufs.size();++i){auto&b=bufs[i];launch[i]={};launch[i].tensorName=b.name.c_str();launch[i].tensorId=ids[i];launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=b.address;if(!b.output)ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"copy input");}
 uint64_t workspace_bytes=0,workspace=0;ck(synWorkspaceGetSize(&workspace_bytes,recipe),"workspace");if(workspace_bytes)ck(synDeviceMalloc(dev,workspace_bytes,0,0,&workspace),"workspace alloc");
 auto run=[&]{ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,0),"launch");};run();
 for(auto&b:bufs)if(b.output)ck(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"copy output");ck(synStreamSynchronize(stream),"correctness sync");for(auto&b:bufs)if(b.output)file(out+"/"+b.name+".bin",b.host,b.bytes,true);
 for(int i=0;i<7;++i)run();ck(synStreamSynchronize(stream),"warmup");synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"event");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"event");
 printf("{\"mode\":\"%s\",\"M\":%u,\"H\":%u,\"workspace_bytes\":%llu}\n",mode.c_str(),m,h,(unsigned long long)workspace_bytes);
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"begin");for(int i=0;i<100;++i)run();ck(synEventRecord(end,stream),"end");ck(synEventSynchronize(end),"sync end");uint64_t ns;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100;printf("{\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,ns/100000.,wall);}
 ck(synEventDestroy(begin),"destroy event");ck(synEventDestroy(end),"destroy event");ck(synStreamDestroy(stream),"destroy stream");ck(synRecipeDestroy(recipe),"recipe destroy");ck(synGraphDestroy(graph),"graph destroy");for(auto&b:bufs){ck(synDeviceFree(dev,b.address,0),"free device");ck(synHostFree(dev,b.host,0),"free host");}if(workspace)ck(synDeviceFree(dev,workspace,0),"free workspace");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");
}
