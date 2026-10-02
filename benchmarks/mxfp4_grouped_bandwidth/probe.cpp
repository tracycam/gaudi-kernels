#include "../../csrc/common/experiment_device.hpp"
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef GK_GROUPED_TILE
#define GK_GROUPED_TILE 256
#endif
static constexpr int Tile=GK_GROUPED_TILE;
static void ck(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static uint16_t bf(float v){uint32_t u;std::memcpy(&u,&v,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static const float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
static void save(const std::string&name,const void*p,size_t n){std::string path=std::string(std::getenv("PROBE_OUT"))+"/"+name;FILE*f=std::fopen(path.c_str(),"wb");if(!f||std::fwrite(p,1,n,f)!=n)throw std::runtime_error("save "+path);std::fclose(f);}
struct Buffer{synTensor t;synSectionHandle section;std::string name;uint64_t bytes,address;void*host;};
#include "manual_ring.hpp"
int main(int argc,char**argv){try{
 if(argc!=10)throw std::invalid_argument("probe tpc|mme|mme-a E pool N K rows splits banks random|mixed|cancel");
 std::string mode=argv[1],pattern=argv[9];int E=std::stoi(argv[2]),pool=std::stoi(argv[3]),N=std::stoi(argv[4]),K=std::stoi(argv[5]),M=std::stoi(argv[6]),S=std::stoi(argv[7]),banks=std::stoi(argv[8]);
 if((mode!="tpc"&&mode!="mme"&&mode!="mme-a")||E<8||E>384||pool!=E*banks||pool>384||N<256||N%Tile||K<32||K%32||(M!=1&&M!=2&&M!=4)||S<1||S>K/32||banks<1||banks>16||(pattern!="random"&&pattern!="mixed"&&pattern!="cancel"&&pattern!="phase"&&pattern!="first-phase"&&pattern!="last-phase"))throw std::invalid_argument("unsupported experiment geometry");
 if(Tile==512&&(mode!="tpc"||M!=1))throw std::invalid_argument("N512 prototype is M1 TPC only");
#ifdef GK_LITERAL_GP
 if(N!=512||K!=6144||S!=3)throw std::invalid_argument("literal GP has N512 K6144 split3");
#endif
 if(pattern=="mixed"&&M!=2)throw std::invalid_argument("mixed means alternating M1/M2");
 uint64_t weight_bytes=uint64_t(pool)*N*K/2,scale_bytes=uint64_t(pool)*N*(K/32),xbytes=uint64_t(E)*M*K*2,ybytes=uint64_t(E)*M*N*4;
 if(weight_bytes+scale_bytes>2ull*1024*1024*1024)throw std::invalid_argument("bounded probe weight limit");
 ck(synInitialize(),"init");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"acquire assigned module");synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");
 std::vector<Buffer>bs;bs.reserve(8);
 auto tensor=[&](std::string name,synDataType type,std::vector<int>dims,uint64_t bytes){synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=dims.size();for(unsigned j=0;j<dims.size();++j)d.m_sizes[j]=d.m_minSizes[j]=dims[j];synSectionHandle sec;ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");synTensor t;ck(synTensorCreate(&t,&d,sec,0),"tensor");Buffer b{t,sec,name,bytes,0,nullptr};ck(synHostMalloc(dev,bytes,0,&b.host),"host alloc");ck(synDeviceMalloc(dev,bytes,0,0,&b.address),"device alloc");std::memset(b.host,0,bytes);bs.push_back(b);return t;};
 auto internal=[&](const char*name,synDataType type,std::vector<int>dims){synTensorDescriptor d{};d.m_name=name;d.m_dataType=type;d.m_dims=dims.size();for(unsigned j=0;j<dims.size();++j)d.m_sizes[j]=d.m_minSizes[j]=dims[j];synTensor t;ck(synTensorCreate(&t,&d,nullptr,0),"transient");return t;};
 auto w=tensor("packed",syn_type_uint8,{Tile/2,K,N/Tile,pool},weight_bytes);auto*wp=(uint8_t*)bs.back().host;
 auto scales=tensor("scales",syn_type_uint8,{Tile,K/32,N/Tile,pool},scale_bytes);auto*sp=(uint8_t*)bs.back().host;
 for(int e=0;e<pool;++e)for(int nb=0;nb<N/Tile;++nb){
  for(int k=0;k<K;++k)for(int j=0;j<Tile/2;++j){auto code=[&](int col){return mix(uint32_t(col*31337+(pattern=="cancel"?k/2:k)*97+e*131))&15;};wp[((uint64_t(e)*(N/Tile)+nb)*K+k)*(Tile/2)+j]=code(nb*Tile+(j/128)*256+j%128)|(code(nb*Tile+(j/128)*256+j%128+128)<<4);}
  for(int g=0;g<K/32;++g)for(int j=0;j<Tile;++j)sp[((uint64_t(e)*(N/Tile)+nb)*(K/32)+g)*Tile+j]=118+mix(((nb*Tile+j)/256)*65537+((nb*Tile+j)%256)*317+g*43+e*103)%15;
 }
 auto lut=tensor("byte_lut",syn_type_bf16,{512,1},1024);auto*lp=(uint16_t*)bs.back().host;for(int i=0;i<256;++i){lp[2*i]=bf(values[i&15]);lp[2*i+1]=bf(values[i>>4]);}
 auto x=tensor("activation",syn_type_bf16,{K,M,E},xbytes);auto*xp=(uint16_t*)bs.back().host;
 auto counts=tensor("counts",syn_type_int32,{E,1},E*4);auto*cp=(int32_t*)bs.back().host;
 int valid_rows=0;
 for(int e=0;e<E;++e){cp[e]=pattern=="mixed"?1+e%2:M;valid_rows+=cp[e];for(int m=0;m<M;++m)for(int k=0;k<K;++k){float a=(int(mix(m*65537+k*131+e*997)%2049)-1024)/1024.f;if(pattern=="cancel")a=(k%2?-1:1)*((m%7)+1)/8.f;if(pattern=="phase"||pattern=="first-phase"||pattern=="last-phase"){a=k%32==e%32?1:0;if(pattern=="first-phase"&&k%2048>=32)a=0;if(pattern=="last-phase"&&k%2048<2016)a=0;}if(m>=cp[e])a=0;xp[(uint64_t(e)*M+m)*K+k]=bf(a);}}
 uint64_t id_stride=((E*4+255)/256)*256;
 auto ids=tensor("expert_ids",syn_type_int32,{E,1},id_stride*banks);auto*ip=(uint8_t*)bs.back().host;
 std::vector<int>permutation(pool);std::iota(permutation.begin(),permutation.end(),0);std::sort(permutation.begin(),permutation.end(),[](int a,int b){return mix(a+3571)<mix(b+3571);});
 for(int b=0;b<banks;++b)for(int e=0;e<E;++e)((int32_t*)(ip+b*id_stride))[e]=permutation[b*E+e];
 auto y=tensor("output",syn_type_single,mode=="mme-a"?std::vector<int>{M,N,E}:std::vector<int>{N,M,E},ybytes);auto*yp=(float*)bs.back().host;
 if(mode=="tpc"){
  auto partial=internal("partial_f32",syn_type_single,{N,M,S,E});synTensor inputs[]={w,scales,lut,x,ids,counts};
#ifdef GK_LITERAL_GP
  auto view=[&](synTensor input,const char*name,synDataType dt,std::vector<int>shape){auto output=internal(name,dt,shape);ck(synNodeCreate(graph,&input,&output,1,1,nullptr,0,"reshape",name,nullptr,nullptr),"literal view");return output;};
  auto fw=view(w,"packed_flat",syn_type_uint8,{256,K*pool}),fs=view(scales,"scales_flat",syn_type_uint8,{512,(K/32)*pool}),fx=view(x,"activation_flat",syn_type_bf16,{K,E});
  auto flat=internal("partial_flat",syn_type_single,{N*S*E,1});synTensor li[]={fw,fs,fx,lut,ids};
  ck(synNodeCreate(graph,li,&flat,5,1,nullptr,0,"gk_grouped_mxfp4_smallm","literal_gp",nullptr,nullptr),"literal grouped node");
  ck(synNodeCreate(graph,&flat,&partial,1,1,nullptr,0,"reshape","partial_view",nullptr,nullptr),"partial view");
#else
  ck(synNodeCreate(graph,inputs,&partial,6,1,nullptr,0,"gk_grouped_mxfp4_smallm","grouped_smallm",nullptr,nullptr),"grouped node");
#endif
  ck(synNodeCreate(graph,&partial,&y,1,1,nullptr,0,S==1?"reshape":"gk_grouped_mxfp4_reduce",S==1?"output_view":"grouped_reduce",nullptr,nullptr),"output");
 }else if(const char* ring=std::getenv("GK_GROUPED_RING_SLOTS")){
  if(mode!="mme"||Tile!=256)throw std::invalid_argument("manual ring supports standard grouped BF16 MME only");
  const char* chunk=std::getenv("GK_GROUPED_RING_CHUNK");
  append_manual_ring(graph,internal,w,scales,lut,ids,x,y,E,N,K,M,std::stoi(ring),chunk?std::stoi(chunk):2,std::getenv("GK_GROUPED_RING_PREFILL")!=nullptr);
 }else{
  auto decoded=internal("decoded_weights",syn_type_bf16,{N,K,E});synTensor inputs[]={w,scales,lut,ids};
  ck(synNodeCreate(graph,inputs,&decoded,4,1,nullptr,0,"gk_grouped_mxfp4_decode","grouped_decode",nullptr,nullptr),"decode");
  synTensor inputs_mme[]={x,decoded};synGEMMParams params{false,false};
  if(mode=="mme-a"){inputs_mme[0]=decoded;inputs_mme[1]=x;params={true,true};}
  ck(synNodeCreate(graph,inputs_mme,&y,2,1,&params,sizeof(params),"batch_gemm","grouped_mme",nullptr,nullptr),"batch MME");
 }
 std::printf("{\"stage\":\"plan\",\"weight_tile\":%d,\"mode\":\"%s\",\"experts\":%d,\"pool\":%d,\"N\":%d,\"K\":%d,\"M_cap\":%d,\"valid_rows\":%d,\"splits\":%d,\"banks\":%d,\"pattern\":\"%s\",\"weight_bytes_per_replay\":%llu,\"rotating_weight_bytes\":%llu,\"activation_bytes\":%llu,\"output_bytes\":%llu}\n",Tile,mode.c_str(),E,pool,N,K,M,valid_rows,S,banks,pattern.c_str(),(unsigned long long)((weight_bytes+scale_bytes)/banks),(unsigned long long)(weight_bytes+scale_bytes),(unsigned long long)xbytes,(unsigned long long)ybytes);std::fflush(stdout);
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"grouped_mxfp4",nullptr),"compile");
 if(std::getenv("GK_MXFP4_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\"}");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;}
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");std::vector<synLaunchTensorInfo> launch;size_t ids_index=0;
 for(auto&b:bs){synLaunchTensorInfo t{};t.tensorName=b.name.c_str();t.tensorType=DATA_TENSOR;t.pTensorAddress=b.address;if(b.name=="expert_ids")ids_index=launch.size();launch.push_back(t);if(b.name!="output")ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"copy");save(b.name+".bin",b.host,b.bytes);}
 uint64_t workspace=0,workspace_bytes=0;ck(synWorkspaceGetSize(&workspace_bytes,recipe),"workspace size");if(workspace_bytes)ck(synDeviceMalloc(dev,workspace_bytes,0,0,&workspace),"workspace alloc");uint64_t ids_base=launch[ids_index].pTensorAddress;
 auto run=[&](int bank){launch[ids_index].pTensorAddress=ids_base+bank*id_stride;ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};
 std::vector<size_t>bad_by_expert(E);size_t checked=0,bad=0;double err2=0,ref2=0,maxabs=0,maxback=0;std::vector<double>reference(uint64_t(E)*M*N);
 for(int bank=0;bank<banks;++bank){
  ck(synMemsetD32Async(bs.back().address,0x7fc12345u,ybytes/4,stream),"poison output");run(bank);ck(synMemCopyAsync(stream,bs.back().address,ybytes,uint64_t(yp),DRAM_TO_HOST),"read output");ck(synStreamSynchronize(stream),"verify sync");save("output_bank"+std::to_string(bank)+".bin",yp,ybytes);
  auto*selected=(int32_t*)(ip+bank*id_stride);
  for(int e=0;e<E;++e)for(int col=0;col<N;++col){int src=selected[e];std::vector<double>weight(K);for(int k=0;k<K;++k){unsigned byte=wp[((uint64_t(src)*(N/Tile)+col/Tile)*K+k)*(Tile/2)+(col%Tile)/256*128+col%128],code=(byte>>((col%256>=128)*4))&15;unsigned exponent=sp[((uint64_t(src)*(N/Tile)+col/Tile)*(K/32)+k/32)*Tile+col%Tile];weight[k]=std::ldexp(double(values[code]),int(exponent)-127);}
   for(int m=0;m<M;++m){double ref=0,absolute=0;if(m<cp[e])for(int k=0;k<K;++k){double product=weight[k]*fp(xp[(uint64_t(e)*M+m)*K+k]);ref+=product;absolute+=std::abs(product);}size_t at=mode=="mme-a"?(uint64_t(e)*N+col)*M+m:(uint64_t(e)*M+m)*N+col;double actual=yp[at],delta=std::abs(actual-ref);reference[at]=ref;bool failed=!std::isfinite(actual)||delta>2e-5+2e-6*absolute;bad+=failed;bad_by_expert[e]+=failed;++checked;err2+=delta*delta;ref2+=ref*ref;maxabs=std::max(maxabs,delta);maxback=std::max(maxback,delta/std::max(absolute,1e-30));}
  }save("oracle_bank"+std::to_string(bank)+".bin",reference.data(),reference.size()*8);
 }
 std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"relative_l2\":%.12g,\"max_abs\":%.12g,\"max_componentwise_backward\":%.12g,\"workspace_bytes\":%llu}\n",checked,bad,std::sqrt(err2/std::max(ref2,1e-30)),maxabs,maxback,(unsigned long long)workspace_bytes);std::fflush(stdout);
 if(bad){std::printf("{\"stage\":\"failure_by_expert\",\"counts\":[");for(int e=0;e<E;++e)std::printf("%s%zu",e?",":"",bad_by_expert[e]);std::puts("]}");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 3;}
 for(int i=0;i<8;++i)run(i%banks);ck(synStreamSynchronize(stream),"warmup");
 synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"event");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"event");int repeats=std::max(20,banks*4);repeats=((repeats+banks-1)/banks)*banks;
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"event begin");for(int i=0;i<repeats;++i)run(i%banks);ck(synEventRecord(end,stream),"event end");ck(synEventSynchronize(end),"event sync");uint64_t ns;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"repeats\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,repeats,double(ns)/1000/repeats,us);}
 ck(synStreamSynchronize(stream),"finish");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
