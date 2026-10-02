#include <synapse_api.h>
#include "common/experiment_device.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
static void ck(synStatus s,const char* action){if(s!=synSuccess){std::fprintf(stderr,"FAIL %s status=%d\n",action,int(s));std::exit(2);}}
static void read(const std::string& path,void* p,size_t bytes){FILE* f=std::fopen(path.c_str(),"rb");if(!f||std::fread(p,1,bytes,f)!=bytes)std::exit(4);std::fclose(f);}
static void write(const std::string& path,const void* p,size_t bytes){FILE* f=std::fopen(path.c_str(),"wb");if(!f||std::fwrite(p,1,bytes,f)!=bytes)std::exit(4);std::fclose(f);}
struct Buffer{synTensor tensor;std::string name;uint64_t bytes,address;void* host;bool input;};
static int probe(synDeviceId device,const std::string& variant,unsigned m,unsigned k,const std::string& fixture,const std::string& output){
 if(!m||!k)return 2;
 const bool lut=variant=="lut_single"||variant=="lut_single4";
 if(!lut&&variant!="baseline"&&variant!="amax16")return 2;
 std::string guid=variant=="baseline"?"fp8_linear_activation_fast":"fp8_fp_quant_"+variant;
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");
 std::vector<Buffer> buffers;buffers.reserve(4);
 auto tensor=[&](const char* name,synDataType dtype,unsigned d0,unsigned d1,unsigned element_bytes,bool input){
  synSectionHandle section;ck(synSectionCreate(&section,0,graph),"section");ck(synSectionSetPersistent(section,true),"persistent");
  synTensorDescriptor d{};d.m_name=name;d.m_dataType=dtype;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;
  Buffer b{nullptr,name,uint64_t(d0)*d1*element_bytes,0,nullptr,input};ck(synTensorCreate(&b.tensor,&d,section,0),"tensor");
  if(dtype==syn_type_fp8_143){synFpQuantParam p{1.,7};synFpQuantMetadata meta{dtype,&p,1};ck(synTensorSetQuantizationData(b.tensor,SYN_FP_QUANT_METADATA,&meta,sizeof(meta)),"fp8 metadata");}
  ck(synHostMalloc(device,b.bytes,0,&b.host),"host allocation");ck(synDeviceMalloc(device,b.bytes,0,0,&b.address),"device allocation");buffers.push_back(b);return b.tensor;
 };
 synTensor x=tensor("activation",syn_type_bf16,k,m,2,true),table=nullptr;
 if(lut)table=tensor("reciprocal_table",syn_type_single,129,1,4,true);
 synTensor q=tensor("native_activation",syn_type_fp8_143,k,m,1,false),s=tensor("activation_scales",syn_type_single,1,m,4,false);
 read(fixture+"/input.bin",buffers[0].host,buffers[0].bytes);if(lut)read(fixture+"/table.bin",buffers[1].host,buffers[1].bytes);
 synTensor inputs[]={x,table},outputs[]={q,s};ck(synNodeCreate(graph,inputs,outputs,lut?2:1,2,nullptr,0,guid.c_str(),"quant_byte_gate",nullptr,nullptr),"quant node");
 synRecipeHandle recipe;std::string recipe_name=output+"/quant_recipe";ck(synGraphCompile(&recipe,graph,recipe_name.c_str(),nullptr),"compile");
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,device,0),"stream");
 std::vector<synLaunchTensorInfo> launch(buffers.size());
 for(size_t i=0;i<buffers.size();++i){auto& b=buffers[i];launch[i]={};launch[i].tensorName=b.name.c_str();launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=b.address;if(b.input)ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"input copy");}
 uint64_t workspace=0,workbytes=0;ck(synWorkspaceGetSize(&workbytes,recipe),"workspace size");if(workbytes)ck(synDeviceMalloc(device,workbytes,0,0,&workspace),"workspace");
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};
 run();for(auto& b:buffers)if(!b.input)ck(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"check sync");
 auto& qb=buffers[buffers.size()-2];auto& sb=buffers.back();write(output+"/q.bin",qb.host,qb.bytes);write(output+"/scale.bin",sb.host,sb.bytes);
 std::vector<uint8_t> expected(qb.bytes);std::vector<uint32_t> expected_scale(m);read(fixture+"/expected-q.bin",expected.data(),expected.size());read(fixture+"/expected-scale.bin",expected_scale.data(),sb.bytes);
 size_t differences=0,scale_differences=0;for(size_t i=0;i<expected.size();++i)differences+=expected[i]!=static_cast<uint8_t*>(qb.host)[i];for(unsigned i=0;i<m;++i)scale_differences+=expected_scale[i]!=static_cast<uint32_t*>(sb.host)[i];
 std::printf("{\"stage\":\"correctness\",\"variant\":\"%s\",\"M\":%u,\"K\":%u,\"checked_bytes\":%zu,\"code_mismatches\":%zu,\"scale_word_mismatches\":%zu,\"workspace_bytes\":%llu}\n",variant.c_str(),m,k,expected.size(),differences,scale_differences,(unsigned long long)workbytes);std::fflush(stdout);
 if(differences||scale_differences)return 3;
 for(int i=0;i<10;++i)run();ck(synStreamSynchronize(stream),"warmup");
 synEventHandle begin,end;ck(synEventCreate(&begin,device,EVENT_COLLECT_TIME),"begin event");ck(synEventCreate(&end,device,EVENT_COLLECT_TIME),"end event");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"record begin");for(int i=0;i<50;++i)run();ck(synEventRecord(end,stream),"record end");ck(synEventSynchronize(end),"event sync");uint64_t ns=0;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/50.;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,ns/50000.,wall);}
 ck(synStreamSynchronize(stream),"finish");return 0;
}

int main(int argc,char**argv){
 if(!std::getenv("PROBE_OUT")||(argc!=3&&argc!=5))return 2;
 ck(synInitialize(),"initialize");synDeviceId device;ck(synDeviceAcquireByModuleId(&device,gaudi_experiment_module()),"explicit module acquire");
 int result=0;
 if(argc==3&&std::string(argv[1])=="batch"){
  std::ifstream file(argv[2]);if(!file)return 4;std::string line;
  while(std::getline(file,line)){std::istringstream row(line);std::string variant,fixture,output;unsigned m,k;
   if(!(row>>variant>>m>>k>>fixture>>output))return 4;
   if(!std::freopen((output+"/run.log").c_str(),"w",stdout))return 4;
   result=probe(device,variant,m,k,fixture,output);std::fflush(stdout);if(result)break;
  }
 }else if(argc==5)result=probe(device,argv[1],std::stoul(argv[2]),std::stoul(argv[3]),argv[4],std::getenv("PROBE_OUT"));
 else result=2;
 ck(synDeviceRelease(device),"release");ck(synDestroy(),"destroy");return result;
}
