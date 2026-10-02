// Synthetic SRAM experiment. No MXFP4 traffic or activation quantization.
#include "../fp8_mme_contract/harness.hpp"
#include <chrono>
#include <set>
int main(int argc,char**argv){try{
 if(argc!=7&&argc!=8)throw std::runtime_error("contention bf16|fp8 resident|pipeline|serial E chunk slots padding_bytes [split-sections]");
 std::string mode=argv[1],schedule=argv[2];int E=std::stoi(argv[3]),chunk=std::stoi(argv[4]),slots=std::stoi(argv[5]);uint64_t padding=std::stoull(argv[6]);
 bool split=argc==8;if(split&&std::string(argv[7])!="split-sections")throw std::runtime_error("bad section mode");
 bool fp8=mode=="fp8";int bytes=fp8?1:2;constexpr int N=512,K=6144,M=2;
 if((mode!="bf16"&&mode!="fp8")||(schedule!="resident"&&schedule!="pipeline"&&schedule!="serial")||(E!=8&&E!=32)||chunk<1||E%chunk||slots<1||slots>3||padding%256)throw std::runtime_error("bad configuration");
 uint64_t tile_bytes=uint64_t(N)*K*chunk*bytes,footprint=slots*tile_bytes+(split?slots*(slots-1)/2:slots-1)*padding;
 if(footprint>16ull*1024*1024||E/chunk<slots)throw std::runtime_error("RMW footprint exceeds 16 MiB");
 Harness h;h.buffers.reserve(256);std::vector<synSectionHandle> sections(split?slots:1);
 for(auto&scratch:sections){check(synSectionCreate(&scratch,0,h.graph),"section");check(synSectionSetPersistent(scratch,false),"scratch transient");check(synSectionSetRMW(scratch,true),"scratch SRAM");}
 auto dt=fp8?syn_type_fp8_143:syn_type_bf16;
 auto weight=[&](int i){std::string name="weight_"+std::to_string(i);synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=dt;d.m_dims=3;d.m_sizes[0]=d.m_minSizes[0]=N;d.m_sizes[1]=d.m_minSizes[1]=K;d.m_sizes[2]=d.m_minSizes[2]=chunk;synTensor t;int slot=i%slots;check(synTensorCreate(&t,&d,sections[split?slot:0],slot*(split?padding:tile_bytes+padding)),"weight");if(fp8){synFpQuantParam p{1.,7};synFpQuantMetadata meta{dt,&p,1};check(synTensorSetQuantizationData(t,SYN_FP_QUANT_METADATA,&meta,sizeof(meta)),"FP8 metadata");}return t;};
 auto node=[&](const char*guid,std::string name,std::vector<synTensor>in,synTensor out,void*params=nullptr,unsigned size=0){synNodeId id;check(synNodeCreateWithId(h.graph,in.data(),&out,in.size(),1,params,size,guid,name.c_str(),&id,nullptr,nullptr),name.c_str());return id;};
 auto depend=[&](synNodeId before,synNodeId after){check(synNodeDependencySet(h.graph,&before,&after,1,1),"explicit dependency");};
 const float values[]={0,.5f,1,1.5f,2,3,4,6};const uint8_t codes[]={0,48,56,60,64,68,72,76};
 auto bf16=[](float v){uint32_t u;std::memcpy(&u,&v,4);return uint16_t(u>>16);};
 std::vector<synTensor>ws;std::vector<synNodeId>gs,last_consumer(slots,0);std::vector<std::vector<float>>refs;std::vector<size_t>output_ids;
 synNodeId previous_g=0,previous_m=0;
 synTensor previous_y=nullptr;
 if(schedule=="serial")previous_y=h.tensor("initial_fence",syn_type_float,{1},true,4);
 auto generate=[&](int i){auto seed=h.tensor("seed_"+std::to_string(i),dt,{N,chunk},true,uint64_t(N)*chunk*bytes);auto host=h.buffers.back().host;
  for(int e=0;e<chunk;++e)for(int n=0;n<N;++n){int code=(n+(i*chunk+e)*3)%8;bool sign=((n/8+i*chunk+e)&1)!=0;if(fp8)((uint8_t*)host)[e*N+n]=codes[code]^(sign?128:0);else((uint16_t*)host)[e*N+n]=bf16(values[code]*(sign?-1.f:1.f));}
  auto w=weight(i);std::vector<synTensor>in={seed};std::string guid=fp8?"gk_generated_feed_fp8":"gk_generated_feed_bf16";
  if(schedule=="serial"){in.push_back(previous_y);guid+="_fenced";}
  auto g=node(guid.c_str(),"generate_"+std::to_string(i),in,w);
  if(previous_g)depend(previous_g,g);previous_g=g;
  if(schedule!="resident"){std::set<synNodeId>before;if(last_consumer[i%slots])before.insert(last_consumer[i%slots]);if(schedule=="serial"&&previous_m)before.insert(previous_m);for(auto b:before)depend(b,g);}
  ws.push_back(w);gs.push_back(g);
 };
 if(schedule=="resident")for(int i=0;i<slots;++i)generate(i);
 for(int i=0;i<E/chunk;++i){
  if(schedule!="resident")generate(i);
  int source=schedule=="resident"?i%slots:i;
  auto a=h.tensor("activation_"+std::to_string(i),dt,{K,M,chunk},true,uint64_t(K)*M*chunk*bytes);auto host=h.buffers.back().host;
  for(int e=0;e<chunk;++e)for(int m=0;m<M;++m)for(int k=0;k<K;++k){int ix=(e*M+m)*K+k;bool sign=((k+i)&1)!=0;if(fp8)((uint8_t*)host)[ix]=(m?64:56)^(sign?128:0);else((uint16_t*)host)[ix]=bf16((m+1)*(sign?-1.f:1.f));}
  auto y=h.tensor("output_"+std::to_string(i),syn_type_float,{N,M,chunk},true,uint64_t(N)*M*chunk*4,true);output_ids.push_back(h.buffers.size()-1);
  refs.emplace_back(N*M*chunk);auto&r=refs.back();for(int e=0;e<chunk;++e)for(int m=0;m<M;++m)for(int n=0;n<N;++n){int code=(n+(source*chunk+e)*3)%8;bool sign=((n/8+source*chunk+e+i)&1)!=0;r[(e*M+m)*N+n]=values[code]*(sign?-1.f:1.f)*K*(m+1);}
  synGEMMParams gp{false,false};auto mme=node("batch_gemm","consume_"+std::to_string(i),{a,ws[source]},y,&gp,sizeof(gp));
  if(schedule=="resident")for(auto g:gs)depend(g,mme);
  last_consumer[i%slots]=mme;previous_m=mme;previous_y=y;
 }
 std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"schedule\":\"%s\",\"E\":%d,\"N\":%d,\"K\":%d,\"M\":%d,\"chunk\":%d,\"slots\":%d,\"padding_bytes\":%llu,\"tile_bytes\":%llu,\"scratch_bytes\":%llu,\"consumed_weight_bytes\":%llu,\"generated_weight_bytes\":%llu,\"real_mxfp4\":false}\n",mode.c_str(),schedule.c_str(),E,N,K,M,chunk,slots,(unsigned long long)padding,(unsigned long long)tile_bytes,(unsigned long long)footprint,(unsigned long long)E*N*K*bytes,(unsigned long long)gs.size()*tile_bytes);std::fflush(stdout);
 std::printf("{\"stage\":\"dependency\",\"serial_data_fence\":%s,\"separate_sections\":%s}\n",schedule=="serial"?"true":"false",split?"true":"false");std::fflush(stdout);
 check(synGraphCompile(&h.recipe,h.graph,"sram_contention",nullptr),"compile");
 if(std::getenv("GK_MXFP4_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\",\"device_launch\":false}");h.close();return 0;}
 h.prepare();h.poison();h.run();h.download("_checked");size_t bad=0,total=0;
 for(size_t j=0;j<refs.size();++j){auto*y=(float*)h.buffers[output_ids[j]].host;for(size_t k=0;k<refs[j].size();++k){++total;bad+=!std::isfinite(y[k])||y[k]!=refs[j][k];}}
 std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu}\n",total,bad);std::fflush(stdout);if(bad)throw std::runtime_error("full exact gate failed");
 for(int i=0;i<8;++i)h.run();check(synStreamSynchronize(h.stream),"warmup");synEventHandle begin,end;check(synEventCreate(&begin,h.dev,EVENT_COLLECT_TIME),"event");check(synEventCreate(&end,h.dev,EVENT_COLLECT_TIME),"event");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();check(synEventRecord(begin,h.stream),"begin");for(int i=0;i<20;++i)h.run();check(synEventRecord(end,h.stream),"end");check(synEventSynchronize(end),"sync");uint64_t ns;check(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,double(ns)/20000,wall);}
 h.close();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"contention: %s\n",e.what());return 1;}}
