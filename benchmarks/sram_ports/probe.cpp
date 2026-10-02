// SRAM endpoint characterization. No model throughput or MXFP4 claim.
#include "../fp8_mme_contract/harness.hpp"
#include <chrono>
#include <algorithm>
#ifndef PORT_WRITE_UNROLL
#define PORT_WRITE_UNROLL 8
#endif
int main(int argc,char**argv){try{
 if(argc!=11)throw std::runtime_error("probe read|write|copy|mme|mix-read|mix-write|mix-copy rows repeats N M K batch mme_nodes a|b|ab|out bf16|fp8");
 std::string mode=argv[1],place=argv[9],dtype=argv[10];
 int rows=std::stoi(argv[2]),reps=std::stoi(argv[3]),N=std::stoi(argv[4]),M=std::stoi(argv[5]),K=std::stoi(argv[6]),B=std::stoi(argv[7]),nodes=std::stoi(argv[8]);
 bool tpc=mode!="mme",mme=mode=="mme"||mode.rfind("mix-",0)==0;
 std::string op=mode.rfind("mix-",0)==0?mode.substr(4):mode;
 bool fp8=dtype=="fp8";int elem=fp8?1:2;
 const char* task_env=std::getenv("GK_PORT_TASKS");int tasks=task_env?std::stoi(task_env):24;
 if(tasks<1||tasks>96)throw std::runtime_error("GK_PORT_TASKS must be 1..96 (tasks, not bound physical cores)");
 bool interleave=std::getenv("GK_PORT_INTERLEAVE");std::vector<int>tpc_shape=interleave?std::vector<int>{64,tasks,rows}:std::vector<int>{64,rows,tasks};
 bool flat=std::getenv("GK_PORT_FLAT");if(flat){if(interleave||mode!="write")throw std::runtime_error("flat experiment supports isolated contiguous writes only");tpc_shape={64,rows*tasks};}
 if(tpc&&op=="write"&&rows%PORT_WRITE_UNROLL)throw std::runtime_error("write rows must be a multiple of the compiled unroll");
 if((tpc&&op!="read"&&op!="write"&&op!="copy")||(place!="a"&&place!="b"&&place!="ab"&&place!="out")||(dtype!="bf16"&&dtype!="fp8")||rows<8||rows%8||reps<1||reps>256||N<1||M<1||K<1||B<1||nodes<1||nodes>128)throw std::runtime_error("bad arguments");
 Harness h;h.buffers.reserve(512);
 synSectionHandle section;check(synSectionCreate(&section,0,h.graph),"section");check(synSectionSetPersistent(section,false),"transient");check(synSectionSetRMW(section,true),"SRAM");
 uint64_t offset=0;
 auto scratch=[&](std::string name,synDataType dt,std::vector<int>shape,uint64_t bytes){
  offset=(offset+255)&~uint64_t(255);if(offset+bytes>16ull*1024*1024)throw std::runtime_error("over 16 MiB scratch budget");
  synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=dt;d.m_dims=shape.size();for(unsigned i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];
  synTensor out;check(synTensorCreate(&out,&d,section,offset),"scratch tensor");offset+=bytes;
  if(dt==syn_type_fp8_143){synFpQuantParam p{1.,7};synFpQuantMetadata q{dt,&p,1};check(synTensorSetQuantizationData(out,SYN_FP_QUANT_METADATA,&q,sizeof(q)),"fp8 meta");}return out;
 };
 auto copy=[&](synTensor a,synTensor b,std::string name){h.node("memcpy",name.c_str(),{a},{b});};
 uint64_t tpc_bytes=uint64_t(64)*rows*tasks*4,ar=uint64_t(M)*K*B*elem,br=uint64_t(N)*K*B*elem,ow=uint64_t(N)*M*B*4;
 size_t tpc_output=0;uint64_t tpc_read=0,tpc_write=0,mme_read=0,mme_write=0;
 synTensor pending_src=nullptr,pending_dst=nullptr,last_mme=nullptr;
 if(tpc){
  synTensor x;
  if(op!="write"){
   auto input=h.tensor("tpc_input",syn_type_uint32,tpc_shape,true,tpc_bytes);
   auto*p=(uint32_t*)h.buffers.back().host;for(uint64_t i=0;i<tpc_bytes/4;++i)p[i]=uint32_t(i%251+1);
   x=scratch("tpc_src",syn_type_uint32,tpc_shape,tpc_bytes);copy(input,x,"tpc_initialize");tpc_read=tpc_bytes*reps;
  }else x=h.tensor("unused_seed",syn_type_uint32,{1},true,4);
  int params[]={rows,reps};synTensor y;
  if(op=="read")y=h.tensor("tpc_result",syn_type_uint32,{64,tasks},true,64*tasks*4,true);
  else{y=scratch("tpc_dst",syn_type_uint32,tpc_shape,tpc_bytes);tpc_write=tpc_bytes*reps;}
  std::string guid="gk_sram_port_"+op+(flat?"_flat":interleave?"_il":"");h.node(guid.c_str(),"port_traffic",{x},{y},params,sizeof(params));
  if(op!="read"){auto out=h.tensor("tpc_result",syn_type_uint32,tpc_shape,true,tpc_bytes,true);if(mme){pending_src=y;pending_dst=out;}else copy(y,out,"tpc_drain");}
  tpc_output=h.buffers.size()-1;
 }
 std::vector<size_t>mmoutputs;
 if(mme){
  auto dt=fp8?syn_type_fp8_143:syn_type_bf16;
  auto operand=[&](std::string name,std::vector<int>shape,uint64_t bytes,bool in_sram,bool is_a){
   auto host=h.tensor(name+"_host",dt,shape,true,bytes);auto*p=h.buffers.back().host;
   uint64_t count=bytes/elem;for(uint64_t i=0;i<count;++i){int value=is_a?1:1+(i%N)%4;if(fp8)((uint8_t*)p)[i]=value==1?56:value==2?64:value==3?68:72;else{float v=float(value);uint32_t bits;std::memcpy(&bits,&v,4);((uint16_t*)p)[i]=bits>>16;}}
   if(!in_sram)return host;auto s=scratch(name,dt,shape,bytes);copy(host,s,name+"_initialize");return s;
  };
  bool as=place=="a"||place=="ab",bs=place=="b"||place=="ab",ys=place=="out";
  auto a=operand("mme_a",{K,M,B},ar,as,true);auto b=operand("mme_b",{N,K,B},br,bs,false);
  synTensor out_sram=nullptr;if(ys){if(nodes!=1)throw std::runtime_error("MME output control requires one node");out_sram=scratch("mme_y",syn_type_float,{N,M,B},ow);}
  for(int i=0;i<nodes;++i){auto out=h.tensor("mme_result_"+std::to_string(i),syn_type_float,{N,M,B},true,ow,true);last_mme=out;mmoutputs.push_back(h.buffers.size()-1);synGEMMParams p{false,false};std::string name="mme_traffic_"+std::to_string(i);h.node("batch_gemm",name.c_str(),{a,b},{ys?out_sram:out},&p,sizeof(p));if(ys)copy(out_sram,out,"mme_drain");}
  mme_read=uint64_t(nodes)*((as?ar:0)+(bs?br:0));mme_write=ys?ow:0;
 }
 if(pending_src){if(interleave)throw std::runtime_error("joined drain requires contiguous task layout");int params[]={rows,1};h.node("gk_sram_port_drain","joined_drain",{pending_src,last_mme},{pending_dst},params,sizeof(params));}
 std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"rows\":%d,\"repeats\":%d,\"tasks\":%d,\"N\":%d,\"M\":%d,\"K\":%d,\"batch\":%d,\"mme_nodes\":%d,\"placement\":\"%s\",\"dtype\":\"%s\",\"scratch_bytes\":%llu,\"tpc_read_bytes\":%llu,\"tpc_write_bytes\":%llu,\"mme_read_bytes\":%llu,\"mme_write_bytes\":%llu,\"scope\":\"logical requested SRAM traffic; setup/drain excluded from byte numerator\"}\n",mode.c_str(),rows,reps,tasks,N,M,K,B,nodes,place.c_str(),dtype.c_str(),(unsigned long long)offset,(unsigned long long)tpc_read,(unsigned long long)tpc_write,(unsigned long long)mme_read,(unsigned long long)mme_write);std::fflush(stdout);
 check(synGraphCompile(&h.recipe,h.graph,"sram_ports",nullptr),"compile");
 if(std::getenv("GK_MXFP4_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\"}");h.close();return 0;}
 h.prepare();h.poison();h.run();h.download("_checked");size_t bad=0,checked=0;
 if(tpc){auto*p=(uint32_t*)h.buffers[tpc_output].host;
  if(op=="read"){for(int t=0;t<tasks;++t)for(int l=0;l<64;++l){uint32_t ref=0;for(int r=0;r<rows;++r){uint64_t index=interleave?(uint64_t(r)*tasks+t)*64+l:(uint64_t(t)*rows+r)*64+l;ref+=uint32_t(index%251+1);}ref*=uint32_t(reps);++checked;bad+=p[t*64+l]!=ref;}}
  else for(uint64_t i=0;i<tpc_bytes/4;++i){uint32_t ref=op=="write"?uint32_t(reps+(interleave?i/64%tasks:i/(64*rows))):uint32_t(i%251+reps);++checked;bad+=p[i]!=ref;}
 }
 for(auto id:mmoutputs){auto*p=(float*)h.buffers[id].host;for(uint64_t i=0;i<ow/4;++i){float ref=K*(1+(i%N)%4);++checked;bad+=p[i]!=ref;}}
 std::printf("{\"stage\":\"correctness\",\"bad\":%zu,\"checked\":%zu}\n",bad,checked);std::fflush(stdout);if(bad)throw std::runtime_error("exact full-output gate failed");
 for(int i=0;i<8;++i)h.run();check(synStreamSynchronize(h.stream),"warmup");synEventHandle begin,end;check(synEventCreate(&begin,h.dev,EVENT_COLLECT_TIME),"event");check(synEventCreate(&end,h.dev,EVENT_COLLECT_TIME),"event");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();check(synEventRecord(begin,h.stream),"begin");for(int i=0;i<20;++i)h.run();check(synEventRecord(end,h.stream),"end");check(synEventSynchronize(end),"wait");uint64_t ns;check(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,double(ns)/20000,wall);}
 h.close();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"sram_ports: %s\n",e.what());return 1;}}
