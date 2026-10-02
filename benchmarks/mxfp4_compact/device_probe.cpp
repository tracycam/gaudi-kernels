#include "../../csrc/ops/mxfp4_compact.hpp"
#include "../../csrc/common/experiment_device.hpp"
#include "../../csrc/tpc/mxfp4_compact/address.h"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace gaudi_kernels::mxfp4_compact;
static void ck(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static uint16_t bf(float x){uint32_t u;std::memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t x){uint32_t u=uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return f;}
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static const float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
static void save(const std::string&name,const void*p,size_t n){std::string path=std::string(std::getenv("PROBE_OUT"))+"/"+name;FILE*f=std::fopen(path.c_str(),"wb");if(!f||std::fwrite(p,1,n,f)!=n)throw std::runtime_error("save "+path);std::fclose(f);}
struct Buffer{synTensor t;synSectionHandle section;std::string name;uint64_t bytes,address;void*host;bool output;};
struct Truth{std::vector<uint8_t>rows,scales;size_t first;};
int main(int argc,char**argv){try{
 if(argc!=9){std::fprintf(stderr,"usage: device_probe mme|decode-native|decode-rows M[,M...] N K random|zero|cancel|edges Ntile bias0|1 repeats\n");return 2;}
 std::string mode=argv[1],pattern=argv[5];int n=std::stoi(argv[3]),k=std::stoi(argv[4]),nt=std::stoi(argv[6]),repeats=std::stoi(argv[8]);bool bias=std::stoi(argv[7]);
 bool decode=mode!="mme",native=mode=="decode-native";
 if((mode!="mme"&&!native&&mode!="decode-rows")||n<1||k<1||(nt!=256&&nt!=512)||repeats<1||repeats>1000)throw std::invalid_argument("bad probe geometry/mode");
 if(pattern!="random"&&pattern!="zero"&&pattern!="cancel"&&pattern!="edges")throw std::invalid_argument("unknown pattern");
 if(!decode&&pattern=="edges")throw std::invalid_argument("MME certificate excludes extreme scales");
 if(native&&(n!=512||k%32))throw std::invalid_argument("native decode gate uses one aligned N512 body");
 if(mode=="decode-rows"&&(n>=512&&k>=32))throw std::invalid_argument("row decoder requires raw row region");
 if(mode=="decode-rows"&&k>=32&&k%32)throw std::invalid_argument("row gate K must be K32 multiple or tail1..31");
 std::vector<int>ms;std::istringstream list(argv[2]);std::string entry;while(std::getline(list,entry,',')){int m=std::stoi(entry);if(m<0)throw std::invalid_argument("negative M");ms.push_back(m);}
 if(ms.empty()||(decode&&(ms.size()!=1||ms[0]!=1)))throw std::invalid_argument("bad M fixture");
 std::printf("{\"stage\":\"acquire\",\"module\":%u}\n",gaudi_experiment_module());std::fflush(stdout);
 ck(synInitialize(),"initialize");synDeviceId device;ck(synDeviceAcquireByModuleId(&device,gaudi_experiment_module()),"assigned module");
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bs;bs.reserve(1+ms.size()*5);
 auto tensor=[&](std::string name,synDataType type,std::vector<int>dims,uint64_t bytes,bool output=false){synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=dims.size();for(unsigned i=0;i<dims.size();++i)d.m_sizes[i]=d.m_minSizes[i]=dims[i];synSectionHandle sec;ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");synTensor t;ck(synTensorCreate(&t,&d,sec,0),"tensor");Buffer b{t,sec,name,bytes,0,nullptr,output};ck(synHostMalloc(device,bytes,0,&b.host),"host alloc");ck(synDeviceMalloc(device,bytes,0,0,&b.address),"device alloc");std::memset(b.host,0,bytes);bs.push_back(b);return t;};
 synTensor lut=tensor("byte_lut",syn_type_bf16,{512,1},1024);auto*table=(uint16_t*)bs.back().host;for(int i=0;i<256;++i){table[2*i]=bf(values[i&15]);table[2*i+1]=bf(values[i>>4]);}
 std::vector<Group>groups;std::vector<Truth>truth;int width=(k+1)/2,ng=(k+31)/32;unsigned smin=255,smax=0;
 for(unsigned e=0;e<ms.size();++e){Group g;g.m=ms[e];g.n=n;g.k=k;g.scales_2_252=true;g.caller_certifies_mme_arithmetic=!decode;Truth t{{},{},bs.size()};if(!g.m){groups.push_back(g);truth.push_back(t);continue;}std::string name="e"+std::to_string(e);
  std::vector<int>wd=decode?(native?std::vector<int>{256,k,1}:std::vector<int>{width,n}):std::vector<int>{n*width};
  std::vector<int>sd=decode?(native?std::vector<int>{512,k/32,1}:std::vector<int>{ng,n}):std::vector<int>{n*ng};
  g.packed=tensor(name+"_packed",syn_type_uint8,wd,uint64_t(n)*width);auto*w=(uint8_t*)bs.back().host;
  g.scales=tensor(name+"_scales",syn_type_uint8,sd,uint64_t(n)*ng);auto*s=(uint8_t*)bs.back().host;
  t.rows.assign(uint64_t(n)*width,0);t.scales.resize(uint64_t(n)*ng);
  for(int col=0;col<n;++col){for(int z=0;z<k;++z){unsigned code=mix(uint32_t(col*31337+(pattern=="cancel"?z/2:z)*97+e*131))&15;if(pattern=="edges")code=(col+z)%16;t.rows[uint64_t(col)*width+z/2]|=code<<(4*(z%2));w[compact_weight_offset(col,z,n,k)]|=code<<compact_nibble_shift(col,z,n,k);}
   // Preserve arbitrary unused checkpoint high nibbles; they must not enter MACs.
   if(k%2){unsigned high=(col%16)<<4;t.rows[uint64_t(col)*width+k/2]|=high;w[compact_weight_offset(col,k-1,n,k)]|=high;}
   for(int group=0;group<ng;++group){unsigned code=118+mix(col*317+group*43+e*103)%15;if(pattern=="edges"){const unsigned edges[]={2,3,126,127,128,251,252};code=edges[(col+group)%7];}t.scales[uint64_t(col)*ng+group]=code;s[compact_scale_offset(col,group,n,k)]=code;smin=std::min(smin,code);smax=std::max(smax,code);}
  }
  save(name+"_checkpoint_rows.bin",t.rows.data(),t.rows.size());save(name+"_checkpoint_scales.bin",t.scales.data(),t.scales.size());
  g.activation=tensor(name+"_activation",syn_type_bf16,{k,g.m},uint64_t(k)*g.m*2);auto*x=(uint16_t*)bs.back().host;
  for(int row=0;row<g.m;++row)for(int z=0;z<k;++z){float a=(int(mix(row*65537+z*131+e*997)%2049)-1024)/1024.f;if(pattern=="zero")a=0;if(pattern=="cancel")a=(z%2?-1:1)*((row%7)+1)/8.f;if(pattern=="cancel"&&k%2&&z==k-1)a=0;x[uint64_t(row)*k+z]=bf(a);}
  std::vector<int>yd=decode?(native?std::vector<int>{n,k}:std::vector<int>{k,n}):std::vector<int>{n,g.m};
  g.output=tensor(name+"_output",decode?syn_type_bf16:syn_type_single,yd,uint64_t(n)*(decode?k:g.m)*(decode?2:4),true);
  if(bias){g.bias=tensor(name+"_bias",syn_type_single,{n,1},uint64_t(n)*4);auto*b=(float*)bs.back().host;for(int col=0;col<n;++col)b[col]=(int(mix(col+e*131)%101)-50)/256.f;}
  groups.push_back(g);truth.push_back(std::move(t));
 }
 Resources resources;Plan plan;plan.native_n_tile=nt;
 if(const char*value=std::getenv("GK_MXFP4_COMPACT_ROW_TILE"))plan.row_n_tile=std::stoi(value);
 if(const char*value=std::getenv("GK_MXFP4_COMPACT_ROW_K_TILE"))plan.row_k_tile=std::stoi(value);
 std::printf("{\"stage\":\"row_tile\",\"row_n_tile\":%d,\"row_k_tile\":%d}\n",plan.row_n_tile,plan.row_k_tile);
 if(decode){synTensor inputs[]={groups[0].packed,groups[0].scales,lut};int params[2]={0,0};ck(synNodeCreate(graph,inputs,&groups[0].output,3,1,native?params:nullptr,native?8:0,native?"gk_mx4c3_native_s2_252_bf16":(plan.row_k_tile==256?"gk_mx4c3_rows256_s2_252_bf16":"gk_mx4c3_rows_s2_252_bf16"),"decode_truth",nullptr,nullptr),"decode node");}
 else resources=append_grouped(graph,lut,groups,plan);
 for(auto&b:bs)save(b.name+".bin",b.host,b.bytes);
 std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"M\":\"%s\",\"N\":%d,\"K\":%d,\"pattern\":\"%s\",\"bias\":%s,\"scale_min\":%u,\"scale_max\":%u,\"scratch_bytes\":%llu,\"original_weight_scale_bytes\":%llu,\"decode_nodes\":%u,\"mme_nodes\":%u,\"native_n_tile\":%d,\"arithmetic_certificate\":\"%s\"}\n",mode.c_str(),argv[2],n,k,pattern.c_str(),bias?"true":"false",smin,smax,(unsigned long long)resources.scratch_bytes,(unsigned long long)resources.original_weight_scale_bytes,resources.decode_nodes,resources.mme_nodes,nt,decode?"decode_only_scales2_252":"fixture_dyadic_activation_integer_over1024_scales118_132_bias_integer_over256_no_subnormal_intermediate");std::fflush(stdout);
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"mxfp4_compact_recovered",nullptr),"compile");
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,device,0),"stream");std::vector<synLaunchTensorInfo>launch;
 for(auto&b:bs){synLaunchTensorInfo l={};l.tensorName=b.name.c_str();l.tensorType=DATA_TENSOR;l.pTensorAddress=b.address;launch.push_back(l);if(!b.output)ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"input copy");}
 uint64_t workspace_bytes=0,workspace=0;ck(synWorkspaceGetSize(&workspace_bytes,recipe),"workspace size");if(workspace_bytes)ck(synDeviceMalloc(device,workspace_bytes,0,0,&workspace),"workspace alloc");
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};run();
 for(auto&b:bs)if(b.output)ck(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"correctness sync");
 size_t checked=0,bad=0;double err2=0,ref2=0,max_abs=0,max_backward=0;
 for(unsigned e=0;e<groups.size();++e){auto&g=groups[e];if(!g.m)continue;auto&t=truth[e];auto*x=(uint16_t*)bs[t.first+2].host;auto&b=bs[t.first+3];auto*bias_ptr=bias?(float*)bs[t.first+4].host:nullptr;save(b.name+".bin",b.host,b.bytes);
  auto weight=[&](int col,int z){unsigned code=(t.rows[uint64_t(col)*width+z/2]>>(4*(z%2)))&15;return std::ldexp(double(values[code]),int(t.scales[uint64_t(col)*ng+z/32])-127);};
  if(decode){std::vector<uint16_t>expected(uint64_t(n)*k);auto*y=(uint16_t*)b.host;for(int z=0;z<k;++z)for(int col=0;col<n;++col){size_t index=native?uint64_t(z)*n+col:uint64_t(col)*k+z;expected[index]=bf(float(weight(col,z)));bad+=expected[index]!=y[index];++checked;}save("e"+std::to_string(e)+"_oracle_bf16.bin",expected.data(),expected.size()*2);}
  else{auto*y=(float*)b.host;std::vector<double>expected(uint64_t(g.m)*n),absolute(expected.size());
   for(int col=0;col<n;++col){std::vector<double>weights(k);for(int z=0;z<k;++z)weights[z]=weight(col,z);for(int row=0;row<g.m;++row){double reference=bias_ptr?bias_ptr[col]:0,scale=std::abs(reference);for(int z=0;z<k;++z){double term=fp(x[uint64_t(row)*k+z])*weights[z];reference+=term;scale+=std::abs(term);}size_t index=uint64_t(row)*n+col;expected[index]=reference;absolute[index]=scale;double error=std::abs(double(y[index])-reference);bool pass=std::isfinite(y[index])&&error<=2e-6*scale;if(pattern=="zero")pass=pass&&double(y[index])==reference;bad+=!pass;++checked;err2+=error*error;ref2+=reference*reference;max_abs=std::max(max_abs,error);max_backward=std::max(max_backward,error/std::max(scale,1e-300));}}
   save("e"+std::to_string(e)+"_oracle_f64.bin",expected.data(),expected.size()*8);save("e"+std::to_string(e)+"_sumabs_f64.bin",absolute.data(),absolute.size()*8);
  }
 }
 std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"relative_l2\":%.12g,\"max_abs\":%.12g,\"max_componentwise_backward\":%.12g,\"workspace_bytes\":%llu}\n",checked,bad,std::sqrt(err2/std::max(ref2,1e-300)),max_abs,max_backward,(unsigned long long)workspace_bytes);std::fflush(stdout);if(bad||!checked)return 3;
 for(int i=0;i<5;++i)run();ck(synStreamSynchronize(stream),"warmup");synEventHandle begin,end;ck(synEventCreate(&begin,device,EVENT_COLLECT_TIME),"event");ck(synEventCreate(&end,device,EVENT_COLLECT_TIME),"event");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"record begin");for(int r=0;r<repeats;++r)run();ck(synEventRecord(end,stream),"record end");ck(synEventSynchronize(end),"event sync");uint64_t ns;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"repeats\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,repeats,double(ns)/1000/repeats,wall);}
 ck(synStreamSynchronize(stream),"finish");ck(synDeviceRelease(device),"release");ck(synDestroy(),"destroy");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
