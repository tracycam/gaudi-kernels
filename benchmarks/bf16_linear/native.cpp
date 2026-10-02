#include "../../csrc/ops/bf16_linear.hpp"
#include "../../csrc/common/experiment_device.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static void ck(synStatus s,const char*w){if(s!=synSuccess){std::fprintf(stderr,"FAIL %s status=%d\n",w,int(s));std::exit(2);}}
struct Buffer{synTensor t;synSectionHandle section;std::string name;uint64_t bytes,address;void*host;};
static void file(const std::string&p,void*data,size_t bytes,bool write){auto*f=std::fopen(p.c_str(),write?"wb":"rb");if(!f||(write?std::fwrite(data,1,bytes,f):std::fread(data,1,bytes,f))!=bytes){std::fprintf(stderr,"FAIL file %s\n",p.c_str());std::exit(4);}std::fclose(f);}
int main(int argc,char**argv){
 if(argc!=10)return 2;unsigned m=std::atoi(argv[2]),n=std::atoi(argv[3]),k=std::atoi(argv[4]);
 std::string mode=argv[1];bool kn=mode=="mme-kn",rowdot=mode=="rowdot"||mode=="rowdot4";
 bool native=mode=="tpc-native",tpc=mode=="tpc"||native,bf16=std::string(argv[5])=="bf16",bias=std::atoi(argv[6]);unsigned split=std::atoi(argv[7]);std::string fixture=argv[8],out=argv[9];
 if(!m||!n||!k)return 2;
 ck(synInitialize(),"initialize");synDeviceId dev;unsigned module=gaudi_experiment_module();ck(synDeviceAcquireByModuleId(&dev,module),"explicit module acquire");
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bufs;bufs.reserve(4);
 auto tensor=[&](const char*name,synDataType dtype,unsigned d0,unsigned d1,uint64_t bytes){synTensorDescriptor d={};d.m_name=name;d.m_dataType=dtype;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;synSectionHandle sec;ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");synTensor t;ck(synTensorCreate(&t,&d,sec,0),"tensor");Buffer b={t,sec,name,bytes,0,nullptr};ck(synHostMalloc(dev,bytes,0,&b.host),"host alloc");ck(synDeviceMalloc(dev,bytes,0,0,&b.address),"device alloc");bufs.push_back(b);return t;};
 synTensor x=tensor("activation_bf16",syn_type_bf16,k,m,uint64_t(k)*m*2);file(fixture+"/activation.bin",bufs.back().host,bufs.back().bytes,false);
 unsigned wn=native?((n+127)/128)*128:n;
 synTensor w=tensor((tpc||kn)?"weight_bf16_kn":"weight_bf16_nk",syn_type_bf16,(tpc||kn)?wn:k,(tpc||kn)?k:n,uint64_t(k)*wn*2);file(fixture+"/weight.bin",bufs.back().host,uint64_t(k)*n*2,false);
 if(tpc||kn){auto*p=(uint16_t*)bufs.back().host;std::vector<uint16_t>copy(p,p+uint64_t(k)*n);for(unsigned j=0;j<wn;++j)for(unsigned z=0;z<k;++z){unsigned mapped=native?(j/128)*128+((j%128)>>1)+((j&1)<<6):j;p[uint64_t(z)*wn+j]=mapped<n?copy[uint64_t(mapped)*k+z]:0;}}
 synTensor b=nullptr;if(bias){b=tensor("bias_fp32",syn_type_single,n,1,uint64_t(n)*4);file(fixture+"/bias.bin",bufs.back().host,bufs.back().bytes,false);}
 synTensor y=tensor("output",bf16?syn_type_bf16:syn_type_single,n,m,uint64_t(n)*m*(bf16?2:4));
 gaudi_kernels::Bf16LinearGraph resources;gaudi_kernels::Bf16LinearOptions options{m,n,k,bf16,tpc,split,native,std::getenv("GK_BF16_SRAM")!=nullptr};options.weight_transposed=kn;options.row_dot=rowdot;options.row_dot_unroll4=mode=="rowdot4"||mode=="hybrid4";options.row_tail=mode=="hybrid"||mode=="hybrid4";
 unsigned depth=std::getenv("GK_BF16_CHAIN_DEPTH")?std::atoi(std::getenv("GK_BF16_CHAIN_DEPTH")):1;
 if(!depth || (depth>1 && (n!=k || !bf16)))return 2;
 synTensor previous=x;
 for(unsigned layer=0;layer<depth;++layer){
  synTensor result=y;
  std::string prefix="layer_"+std::to_string(layer);options.name_prefix=prefix.c_str();
  if(layer+1<depth){synTensorDescriptor d={};d.m_name=prefix.c_str();d.m_dataType=syn_type_bf16;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=n;d.m_sizes[1]=d.m_minSizes[1]=m;ck(synTensorCreate(&result,&d,nullptr,0),"chain tensor");resources.intermediates.push_back(result);}
  ck(gaudi_kernels::add_bf16_linear(graph,previous,w,b,result,options,&resources),"add bf16 linear");previous=result;
 }
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"bf16_linear",nullptr),"compile");
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");std::vector<synLaunchTensorInfo>launch(bufs.size());
 for(size_t i=0;i<bufs.size();++i){auto&v=bufs[i];launch[i]={};launch[i].tensorName=v.name.c_str();launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=v.address;if(i+1<bufs.size())ck(synMemCopyAsync(stream,uint64_t(v.host),v.bytes,v.address,HOST_TO_DRAM),"input copy");}
 uint64_t workspaceBytes=0,workspace=0;ck(synWorkspaceGetSize(&workspaceBytes,recipe),"workspace size");if(workspaceBytes)ck(synDeviceMalloc(dev,workspaceBytes,0,0,&workspace),"workspace alloc");
 bool launch_ids=std::getenv("GK_BF16_LAUNCH_IDS")!=nullptr;
 if(launch_ids){std::vector<const char*>names;std::vector<uint64_t>ids(bufs.size());for(auto&v:bufs)names.push_back(v.name.c_str());ck(synTensorRetrieveIds(recipe,names.data(),ids.data(),ids.size()),"resolve tensor IDs once");for(size_t i=0;i<ids.size();++i)launch[i].tensorId=ids[i];}
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,launch_ids?0:SYN_FLAGS_TENSOR_NAME),"launch");};
 run();auto&output=bufs.back();ck(synMemCopyAsync(stream,output.address,output.bytes,uint64_t(output.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"result sync");file(out+"/output.bin",output.host,output.bytes,true);
 std::printf("{\"stage\":\"layout\",\"module\":%u,\"workspace_bytes\":%llu,\"weight_bytes\":%llu,\"offline_weight_transpose\":%s,\"intermediate_tensors\":%zu}\n",module,(unsigned long long)workspaceBytes,(unsigned long long)(uint64_t(k)*wn*2),(tpc||kn)?"true":"false",resources.intermediates.size());std::fflush(stdout);
 for(int i=0;i<6;++i)run();ck(synStreamSynchronize(stream),"warmup");synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"event");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"event");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"begin");for(int i=0;i<50;++i)run();ck(synEventRecord(end,stream),"end");ck(synEventSynchronize(end),"event sync");uint64_t ns=0;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/50;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,ns/50000.,us);}
 ck(synStreamSynchronize(stream),"finish");ck(synEventDestroy(begin),"event destroy");ck(synEventDestroy(end),"event destroy");ck(synStreamDestroy(stream),"stream destroy");ck(synRecipeDestroy(recipe),"recipe destroy");ck(synGraphDestroy(graph),"graph destroy");
 if(workspace)ck(synDeviceFree(dev,workspace,0),"workspace free");for(auto&v:bufs){ck(synDeviceFree(dev,v.address,0),"device free");ck(synHostFree(dev,v.host,0),"host free");}
 ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;
}
