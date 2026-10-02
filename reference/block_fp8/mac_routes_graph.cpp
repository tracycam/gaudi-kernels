// Original one-byte HBM payload, BF16 or native FP8 MME, FP32 output.
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static void ck(synStatus s,const char*where){if(s!=synSuccess){std::fprintf(stderr,"FAIL %s status=%d\n",where,int(s));std::exit(2);}}
static uint16_t bf(float v){uint32_t u;std::memcpy(&u,&v,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static float ocp(uint8_t v){int e=(v>>3)&15,m=v&7;float x=e?std::ldexp(1.f+m/8.f,e-7):m/512.f;if(e==15&&m==7)x=NAN;return v&128?-x:x;}
static float quant(float v){float a=std::min(448.f,std::abs(v));int e=int(std::floor(std::log2(std::max(a,1.f/64.f))))-3;float step=std::ldexp(1.f,e);return std::copysign(std::nearbyint(a/step)*step,v);}
static void load_fixture(const char*name,void*dest,size_t bytes){auto*dir=std::getenv("ORIGINAL_FP8_FIXTURE");if(!dir)return;std::string path=std::string(dir)+"/"+name;auto*f=std::fopen(path.c_str(),"rb");if(!f||std::fread(dest,1,bytes,f)!=bytes){std::fprintf(stderr,"FAIL fixture %s\n",path.c_str());std::exit(4);}std::fclose(f);}
struct Buffer{synTensor t;synSectionHandle section;std::string name;uint64_t bytes,address;void*host;};
int main(int argc,char**argv){
 if(argc!=5)return 2;
 std::string mode=argv[1];bool native=mode=="native"||mode=="adaptive";
 bool output_bf16=std::getenv("MAC_OUTPUT_BF16")!=nullptr;bool adaptive=mode=="adaptive";bool prepared=native||mode=="bf16-native"||mode=="bf16-fast";bool decode=false,packed=true,block=false;int m=std::atoi(argv[2]),n=std::atoi(argv[3]),k=std::atoi(argv[4]);
 if(m<1||n<1||k<1||k%128||(!prepared&&mode!="bf16"))return 2;int groups=k/128;
 ck(synInitialize(),"initialize");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,5),"module5");
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bufs;bufs.reserve(4);
 int nt=std::getenv("MAC_NT")?std::atoi(std::getenv("MAC_NT")):n;
 bool rmw=std::getenv("MAC_RMW")!=nullptr;
 uint64_t activation_capacity=(uint64_t(m)*k+255)&~255ull;
 uint64_t scale_capacity=(uint64_t(m)*groups*4+255)&~255ull;
 uint64_t rmw_limit=0;synDeviceAttribute attribute=DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
 ck(synDeviceGetAttribute(&rmw_limit,&attribute,1,dev),"RMW capacity");
 if(rmw&&!std::getenv("MAC_NT")){
  if(rmw_limit<=activation_capacity+scale_capacity)return 2;
  uint64_t fit=(rmw_limit-activation_capacity-scale_capacity)/(uint64_t(m)*groups*4);
  if(fit<uint64_t(n))nt=int(fit/128)*128;
 }
 nt=std::min(n,nt);if(nt<1||(nt<n&&nt%128))return 2;
 uint64_t partial_capacity=(uint64_t(nt)*m*groups*4+255)&~255ull;
 std::printf("{\"stage\":\"layout\",\"N_tile\":%d,\"rmw_enabled\":%s,\"rmw_limit_bytes\":%llu,\"rmw_requested_bytes\":%llu}\n",nt,rmw?"true":"false",(unsigned long long)rmw_limit,(unsigned long long)(rmw?partial_capacity+activation_capacity+scale_capacity:0));std::fflush(stdout);
 if(rmw&&partial_capacity+activation_capacity+scale_capacity>rmw_limit)return 2;
 synSectionHandle shared_rmw=nullptr;
 if(rmw){ck(synSectionCreate(&shared_rmw,0,graph),"shared RMW");ck(synSectionSetPersistent(shared_rmw,false),"RMW nonpersistent");ck(synSectionSetRMW(shared_rmw,true),"RMW enable");}
 auto tensor=[&](const char*name,synDataType dtype,int d0,int d1,bool persistent,uint64_t bytes,int d2=0){
  synTensorDescriptor d={};d.m_name=name;d.m_dataType=dtype;d.m_dims=d2?3:2;
  d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;
  if(d2)d.m_sizes[2]=d.m_minSizes[2]=d2;
  synSectionHandle sec=nullptr;if(persistent){ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");}
  else if(rmw&&(std::string(name).find("block_partials")==0||std::string(name)=="activation_scales"||std::string(name)=="native_activation"))sec=shared_rmw;
  uint64_t offset=0;
  if(sec==shared_rmw&&rmw){if(std::string(name)=="native_activation")offset=partial_capacity;else if(std::string(name)=="activation_scales")offset=partial_capacity+activation_capacity;}
  synTensor t;ck(synTensorCreate(&t,&d,sec,offset),"tensor");
  if(dtype==syn_type_fp8_143){synFpQuantParam param{1.0,7};synFpQuantMetadata meta{dtype,&param,1};ck(synTensorSetQuantizationData(t,SYN_FP_QUANT_METADATA,&meta,sizeof(meta)),"fp8 metadata");}
  if(persistent){Buffer b={t,sec,name,bytes,0,nullptr};ck(synHostMalloc(dev,bytes,0,&b.host),"host alloc");ck(synDeviceMalloc(dev,bytes,0,0,&b.address),"device alloc");bufs.push_back(b);}return t;
 };
 synTensor w=tensor(prepared?"prepared_native_fp8_bytes":"original_fp8_bytes",prepared?syn_type_fp8_143:syn_type_uint8,128,n,true,uint64_t(k)*n,groups);
 auto* wc=static_cast<uint8_t*>(bufs[0].host);
 for(uint64_t i=0;i<uint64_t(n)*k;++i){int code=decode?i%256:(i*73+17)%254;if(!decode&&code>=127)++code;wc[i]=code;}
 load_fixture("weight.bin",wc,uint64_t(n)*k);
 std::vector<uint8_t>original_weight;
 if(packed){original_weight.assign(wc,wc+uint64_t(n)*k);for(int row=0;row<n;++row)for(int z=0;z<k;++z)wc[(uint64_t(z/128)*n+row)*128+z%128]=original_weight[uint64_t(row)*k+z];}
 const uint8_t* oracle_weight=packed?original_weight.data():wc;
 synTensor x=nullptr;if(!decode){x=tensor("activation_bf16",syn_type_bf16,k,m,true,uint64_t(k)*m*2);auto*p=static_cast<uint16_t*>(bufs[1].host);for(uint64_t i=0;i<uint64_t(m)*k;++i)p[i]=bf(float(int((i*97+3)%257)-128)/128.f);}
 if(!decode)load_fixture("activation.bin",bufs[1].host,uint64_t(k)*m*2);
 synTensor ws=tensor("original_block_scales",syn_type_single,groups,(n+127)/128,true,uint64_t(groups)*((n+127)/128)*4);
 auto*scales=static_cast<float*>(bufs[2].host);for(int i=0;i<groups*((n+127)/128);++i)scales[i]=float(1+i%17)*.0001f;
 load_fixture("scales.bin",scales,bufs[2].bytes);
 std::vector<float> original_scales(scales,scales+groups*((n+127)/128));
 if(prepared){
  // Original checkpoints remain untouched. This one-byte device layout is
  // prepared once, with RNE for the small values affected by halving.
  for(int nb=0;nb<(n+127)/128;++nb)for(int g=0;g<groups;++g){
   bool half=!adaptive;
   if(adaptive)for(int row=nb*128;row<std::min(n,nb*128+128);++row)for(int z=0;z<128;++z)half|=(original_weight[uint64_t(row)*k+g*128+z]&127)>119;
   if(!half)continue;
   scales[nb*groups+g]*=2;
   for(int row=nb*128;row<std::min(n,nb*128+128);++row)for(int z=0;z<128;++z){
    uint64_t i=(uint64_t(g)*n+row)*128+z;unsigned c=wc[i],mag=c&127;
    unsigned h=mag>=16?mag-8:(mag>>1)+((mag&1)&&((mag>>1)&1));wc[i]=(c&128)|h;
   }
  }
 }
 synTensor y=tensor("output_f32",output_bf16?syn_type_bf16:syn_type_single,n,m,true,uint64_t(n)*m*(output_bf16?2:4));
 synGEMMParams params{false,true};
 if(native){
  synTensor a=tensor("native_activation",syn_type_fp8_143,128,m,false,0,groups);
  synTensor sa=tensor("activation_scales",syn_type_single,1,m,false,0,groups);
  synTensor qo[]={a,sa};ck(synNodeCreate(graph,&x,qo,1,2,nullptr,0,"mac_activation_native","activation_quant",nullptr,nullptr),"activation quant");
  int count=(n+nt-1)/nt;std::vector<synTensor> weights,weight_scales,results;
  for(int tile=0;tile<count;++tile){int size=std::min(nt,n-tile*nt);std::string suffix=std::to_string(tile);
   weights.push_back(count==1?w:tensor(("weight_tile_"+suffix).c_str(),syn_type_fp8_143,128,size,false,0,groups));
   weight_scales.push_back(count==1?ws:tensor(("weight_scale_tile_"+suffix).c_str(),syn_type_single,groups,(size+127)/128,false,0));
   results.push_back(count==1?y:tensor(("output_tile_"+suffix).c_str(),output_bf16?syn_type_bf16:syn_type_single,size,m,false,0));
  }
  if(count>1){synSplitParams split{1};ck(synNodeCreate(graph,&w,weights.data(),1,count,&split,sizeof(split),"split","split_weights",nullptr,nullptr),"weight split");
   ck(synNodeCreate(graph,&ws,weight_scales.data(),1,count,&split,sizeof(split),"split","split_scales",nullptr,nullptr),"scale split");}
  synNodeId previous=0;
  for(int tile=0;tile<count;++tile){int size=std::min(nt,n-tile*nt);std::string suffix=std::to_string(tile);
   synTensor p=tensor(("block_partials_"+suffix).c_str(),syn_type_single,size,m,false,0,groups);
   synTensor inputs[]={a,weights[tile]};synNodeId producer,consumer;
   ck(synNodeCreateWithId(graph,inputs,&p,2,1,&params,sizeof(params),"batch_gemm",("native_mme_"+suffix).c_str(),&producer,nullptr,nullptr),"native mme");
   synTensor ri[]={p,sa,weight_scales[tile]};const char*guid=std::getenv("MAC_REDUCE_ROWS")&&std::atoi(std::getenv("MAC_REDUCE_ROWS"))==4?"mac_block_reduce4":"mac_block_reduce1";
   std::string output_guid=std::string(guid)+(output_bf16?"_bf16":"");
   ck(synNodeCreateWithId(graph,ri,&results[tile],3,1,nullptr,0,output_guid.c_str(),("block_scale_reduce_"+suffix).c_str(),&consumer,nullptr,nullptr),"reduce");
   if(tile&&rmw)ck(synNodeDependencySet(graph,&previous,&producer,1,1),"shared RMW reuse dependency");previous=consumer;
  }
  if(count>1){synConcatenateParams concat{0};ck(synNodeCreate(graph,results.data(),&y,count,1,&concat,sizeof(concat),"concat","join_outputs",nullptr,nullptr),"output concat");}

 }else{
  synTensor d=tensor("decoded_scaled_bf16",syn_type_bf16,k,n,false,0);
  synTensor di[]={w,ws};ck(synNodeCreate(graph,di,&d,2,1,nullptr,0,prepared?(mode=="bf16-fast"?"mac_native_scaled_bf16_fast":"mac_native_scaled_bf16"):"mac_original_scaled_bf16","weight_decode",nullptr,nullptr),"decode");
  int ms=std::getenv("MAC_M_SPLIT")?std::atoi(std::getenv("MAC_M_SPLIT")):0;
  if(ms){
   if(ms<1||ms>=m)return 2;
   synTensor xs[]={tensor("activation_main",syn_type_bf16,k,ms,false,0),tensor("activation_tail",syn_type_bf16,k,m-ms,false,0)};
   synDataType dtype=output_bf16?syn_type_bf16:syn_type_single;
   synTensor ys[]={tensor("output_main",dtype,n,ms,false,0),tensor("output_tail",dtype,n,m-ms,false,0)};
   synSplitParams split{1};ck(synNodeCreate(graph,&x,xs,1,2,&split,sizeof(split),"split","split_activation_rows",nullptr,nullptr),"activation row split");
   for(int part=0;part<2;++part){synTensor mi[]={xs[part],d};ck(synNodeCreate(graph,mi,&ys[part],2,1,&params,sizeof(params),"gemm",part?"bf16_mme_tail":"bf16_mme_main",nullptr,nullptr),"split bf16 mme");}
   synConcatenateParams concat{1};ck(synNodeCreate(graph,ys,&y,2,1,&concat,sizeof(concat),"concat","join_output_rows",nullptr,nullptr),"output row concat");
  }else{
   synTensor mi[]={x,d};ck(synNodeCreate(graph,mi,&y,2,1,&params,sizeof(params),"gemm","bf16_mme",nullptr,nullptr),"bf16 mme");
  }
 }
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"mac_routes",nullptr),"compile");
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");
 std::vector<synLaunchTensorInfo>launch(bufs.size());
 for(size_t i=0;i<bufs.size();++i){auto&b=bufs[i];launch[i]={};launch[i].tensorName=b.name.c_str();launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=b.address;if(i+1<bufs.size())ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"input copy");}
 uint64_t workspaceBytes=0,workspace=0;ck(synWorkspaceGetSize(&workspaceBytes,recipe),"workspace size");if(workspaceBytes)ck(synDeviceMalloc(dev,workspaceBytes,0,0,&workspace),"workspace alloc");
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};
 run();auto&output=bufs.back();ck(synMemCopyAsync(stream,output.address,output.bytes,uint64_t(output.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"sync");
 if(const char*path=std::getenv("ORIGINAL_FP8_OUTPUT")){auto*f=std::fopen(path,"wb");if(!f||std::fwrite(output.host,1,output.bytes,f)!=output.bytes)return 4;std::fclose(f);}
 size_t checked=0,bad=0;double error=0,energy=0,maxerror=0;
 if(decode){auto*actual=static_cast<uint16_t*>(output.host);for(uint64_t i=0;i<uint64_t(n)*k;++i){float ref=ocp(wc[i]);bool ok=std::isnan(ref)?std::isnan(fp(actual[i])):actual[i]==bf(ref);bad+=!ok;++checked;}}
 else{auto*xc=static_cast<uint16_t*>(bufs[1].host);auto*actual=static_cast<float*>(output.host);
  for(int row:std::vector<int>{0,m/2,m-1})for(int j=0;j<std::min(n,64);++j){int col=j*(n-1)/std::max(1,std::min(n,64)-1);double ref=0;for(int z=0;z<k;++z){double a=double(fp(xc[uint64_t(row)*k+z]));double weight=ocp(oracle_weight[uint64_t(col)*k+z]);weight*=original_scales[(col/128)*groups+z/128];ref+=a*weight;}double a=output_bf16?double(fp(static_cast<uint16_t*>(output.host)[uint64_t(row)*n+col])):actual[uint64_t(row)*n+col],e=a-ref;bad+=!std::isfinite(a);error+=e*e;energy+=ref*ref;maxerror=std::max(maxerror,std::abs(e));++checked;}
 }
 double relative=std::sqrt(error/std::max(energy,1e-30));if(relative>0.15)++bad; // broad smoke check; independent FP64 MAC report decides quality
 std::printf("{\"stage\":\"correctness\",\"decode_only\":%s,\"M\":%d,\"N\":%d,\"K\":%d,\"checked\":%zu,\"bad\":%zu,\"relative_l2\":%.12g,\"max_abs\":%.12g,\"workspace_bytes\":%llu}\n",decode?"true":"false",m,n,k,checked,bad,relative,maxerror,(unsigned long long)workspaceBytes);std::fflush(stdout);if(bad)return 3;
 for(int i=0;i<5;++i)run();ck(synStreamSynchronize(stream),"warmup");
 synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"begin event");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"end event");
 for(int sample=0;sample<5;++sample){auto wall=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"event begin");for(int i=0;i<20;++i)run();ck(synEventRecord(end,stream),"event end");ck(synEventSynchronize(end),"event sync");uint64_t ns=0;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-wall).count()/20;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,ns/20000.,us);}
 ck(synStreamSynchronize(stream),"finish");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;
}
