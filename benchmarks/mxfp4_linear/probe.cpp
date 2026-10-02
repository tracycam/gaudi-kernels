#include "../../csrc/ops/mxfp4_linear.hpp"
#include "../../csrc/common/experiment_device.hpp"
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
using namespace gaudi_kernels::mxfp4;
static void ck(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static uint16_t bf(float v){uint32_t u;std::memcpy(&u,&v,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static const float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
struct Buffer {synTensor t;synSectionHandle sec;std::string name;uint64_t bytes,address;void*host;bool output;};
static void save(const std::string&name,const void*p,size_t n){const char*dir=std::getenv("PROBE_OUT");if(!dir)return;std::string path=std::string(dir)+"/"+name;FILE*f=std::fopen(path.c_str(),"wb");if(!f||std::fwrite(p,1,n,f)!=n)throw std::runtime_error("save "+path);std::fclose(f);}
int main(int argc,char**argv){try{
 if(argc!=8){std::fprintf(stderr,"usage: probe mme|tpc|decode M[,M...] N K random|zero|cancel|impulse|edges Ntile bias0|1\n");return 2;}
 std::string mode=argv[1],pattern=argv[5];int n=std::stoi(argv[3]),k=std::stoi(argv[4]),nt=std::stoi(argv[6]);bool bias=std::stoi(argv[7]);
 if((mode!="mme"&&mode!="tpc"&&mode!="decode")||n<1||k<1||nt<256||nt%256)throw std::invalid_argument("invalid mode/shape");
 std::vector<int> ms;std::istringstream list(argv[2]);std::string entry;while(std::getline(list,entry,',')){int m=std::stoi(entry);if(m<0)throw std::invalid_argument("negative M");ms.push_back(m);}if(ms.empty())return 2;
 if(mode=="decode"&&(ms.size()!=1||ms[0]!=1))throw std::invalid_argument("decode has one fixture");
 ck(synInitialize(),"initialize");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"assigned module");
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bs;bs.reserve(1+ms.size()*5);
 auto tensor=[&](std::string name,synDataType dtype,std::vector<int>dims,uint64_t bytes,bool output=false){synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=dtype;d.m_dims=dims.size();for(size_t i=0;i<dims.size();++i)d.m_sizes[i]=d.m_minSizes[i]=dims[i];synSectionHandle s;ck(synSectionCreate(&s,0,graph),"section");ck(synSectionSetPersistent(s,true),"persistent");synTensor t;ck(synTensorCreate(&t,&d,s,0),"tensor");Buffer b{t,s,name,bytes,0,nullptr,output};ck(synHostMalloc(dev,bytes,0,&b.host),"host alloc");ck(synDeviceMalloc(dev,bytes,0,0,&b.address),"device alloc");std::memset(b.host,0,bytes);bs.push_back(b);return t;};
 synTensor lut=tensor("byte_lut",syn_type_bf16,{512,1},1024);auto*table=(uint16_t*)bs.back().host;for(int i=0;i<256;++i){table[2*i]=bf(values[i&15]);table[2*i+1]=bf(values[i>>4]);}
 std::vector<Group> groups;std::vector<size_t> starts;unsigned minscale=255,maxscale=0;int nb=(n+255)/256,ng=(k+31)/32;
 for(size_t e=0;e<ms.size();++e){Group g;g.m=ms[e];g.n=n;g.k=k;starts.push_back(bs.size());if(!g.m){groups.push_back(g);continue;}std::string tag="e"+std::to_string(e);
  g.packed=tensor(tag+"_packed",syn_type_uint8,{128,k,nb},uint64_t(nb)*k*128);auto*w=(uint8_t*)bs.back().host;
  g.scales=tensor(tag+"_e8m0",syn_type_uint8,{256,ng,nb},uint64_t(nb)*ng*256);auto*s=(uint8_t*)bs.back().host;
  for(int b=0;b<nb;++b)for(int z=0;z<k;++z)for(int j=0;j<128;++j){auto code=[&](int col){if(col>=n)return 0u;if(pattern=="exhaustive")return unsigned(col%16);return mix(uint32_t(col*31337+(pattern=="cancel"?z/2:z)*97+e*131))&15;};w[(uint64_t(b)*k+z)*128+j]=code(b*256+j)|(code(b*256+j+128)<<4);}
  for(int b=0;b<nb;++b)for(int z=0;z<ng;++z)for(int j=0;j<256;++j){unsigned exponent=118+mix(b*65537+j*317+z*43+e*103)%15;if(pattern=="exhaustive")exponent=2+(b*256+j)/16%251;if(pattern=="edges"){const unsigned edges[]={2,3,126,127,128,251,252};exponent=edges[(j+z)%7];}if(exponent<2||exponent>252)throw std::runtime_error("unsupported E8M0");s[(uint64_t(b)*ng+z)*256+j]=exponent;minscale=std::min(minscale,exponent);maxscale=std::max(maxscale,exponent);}
  g.activation=tensor(tag+"_activation",syn_type_bf16,{k,g.m},uint64_t(k)*g.m*2);auto*x=(uint16_t*)bs.back().host;
  for(int row=0;row<g.m;++row)for(int z=0;z<k;++z){float a=(int(mix(row*65537+z*131+e*997)%2049)-1024)/1024.f;if(pattern=="zero")a=0;if(pattern=="cancel")a=(z%2?-1:1)*((row%7)+1)/8.f;if(pattern=="impulse")a=z==(row*17)%k?1:0;x[uint64_t(row)*k+z]=bf(a);}
  g.output=tensor(tag+"_output",mode=="decode"?syn_type_bf16:syn_type_single,{n,mode=="decode"?k:g.m},uint64_t(n)*(mode=="decode"?k:g.m)*(mode=="decode"?2:4),true);
  if(bias){g.bias=tensor(tag+"_bias",syn_type_single,{n,1},uint64_t(n)*4);auto*p=(float*)bs.back().host;for(int j=0;j<n;++j)p[j]=(int(mix(j+e*131)%101)-50)/256.f;}
  groups.push_back(g);
 }
 Plan plan;if(const char* b=std::getenv("GK_MXFP4_SCRATCH_BUFFERS"))plan.scratch_buffers=std::stoi(b);
 if(const char* b=std::getenv("GK_MXFP4_UNIFIED_SCRATCH")){int v=std::stoi(b);if(v!=0&&v!=1)throw std::invalid_argument("unified scratch must be 0 or 1");plan.unified_scratch_section=v;}
 if(const char* b=std::getenv("GK_MXFP4_PREFETCH_PACKED")){int v=std::stoi(b);if(v!=0&&v!=1)throw std::invalid_argument("packed prefetch must be 0 or 1");plan.prefetch_packed_to_sram=v;}
 if(const char* b=std::getenv("GK_MXFP4_LOGICAL_WEIGHT_MIB")){
  int mib=std::stoi(b);if(!std::getenv("GK_PIPELINE_COMPILER_SRAM")||mib<1||mib>512)throw std::invalid_argument("logical weight budget is private compiler-managed diagnostic only");
  plan.scratch_limit=uint64_t(mib)*1024*1024;
 }
 plan.n_tile=nt;plan.tpc_gemv=mode=="tpc";Resources resources;
 if(mode=="decode"){synTensor in[]={groups[0].packed,groups[0].scales,lut};int offset=0;ck(synNodeCreate(graph,in,&groups[0].output,3,1,&offset,4,"gk_mxfp4_decode_bf16_v1","decode_truth",nullptr,nullptr),"decode truth");}
 else resources=append_grouped(graph,lut,groups,plan);
 std::printf("{\"stage\":\"scratch_layout\",\"buffers\":%u,\"unified_section\":%s}\n",plan.scratch_buffers,plan.unified_scratch_section?"true":"false");
 std::printf("{\"stage\":\"prefetch\",\"packed_to_sram\":%s,\"dma_nodes\":%u}\n",plan.prefetch_packed_to_sram?"true":"false",resources.dma_nodes);
 std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"M\":\"%s\",\"N\":%d,\"K\":%d,\"pattern\":\"%s\",\"bias\":%s,\"scale_min\":%u,\"scale_max\":%u,\"scratch_bytes\":%llu,\"logical_weight_bytes\":%llu,\"decode_nodes\":%u,\"mme_nodes\":%u,\"empty_groups\":%u,\"n_tile\":%d}\n",mode.c_str(),argv[2],n,k,pattern.c_str(),bias?"true":"false",minscale,maxscale,(unsigned long long)resources.scratch_bytes,(unsigned long long)resources.logical_weight_bytes,resources.decode_nodes,resources.mme_nodes,resources.empty_groups,nt);std::fflush(stdout);
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"mxfp4_coverage",nullptr),"compile");
 if(std::getenv("GK_MXFP4_COMPILE_ONLY")){
  std::puts("{\"stage\":\"compile_only\",\"kernel_launched\":false}");
  ck(synDeviceRelease(dev),"compile-only release");ck(synDestroy(),"compile-only destroy");return 0;
 }
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,dev,0),"stream");std::vector<synLaunchTensorInfo> launch;
 for(auto&b:bs){synLaunchTensorInfo l={};l.tensorName=b.name.c_str();l.tensorType=DATA_TENSOR;l.pTensorAddress=b.address;launch.push_back(l);if(!b.output)ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"input copy");save(b.name+".bin",b.host,b.bytes);}
 uint64_t workspaceBytes=0,workspace=0;ck(synWorkspaceGetSize(&workspaceBytes,recipe),"workspace size");if(workspaceBytes)ck(synDeviceMalloc(dev,workspaceBytes,0,0,&workspace),"workspace alloc");
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};run();
 for(auto&b:bs)if(b.output)ck(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"correctness sync");
 size_t checked=0,bad=0;double err2=0,ref2=0,maxabs=0,maxback=0;std::vector<double> refs;
 for(size_t e=0;e<groups.size();++e){auto&g=groups[e];if(!g.m)continue;size_t at=starts[e];auto*w=(uint8_t*)bs[at].host;auto*s=(uint8_t*)bs[at+1].host;auto*x=(uint16_t*)bs[at+2].host;auto&out=bs[at+3];auto*b=bias?(float*)bs[at+4].host:nullptr;save(out.name+".bin",out.host,out.bytes);
  auto weight=[&](int col,int z){unsigned byte=w[(uint64_t(col/256)*k+z)*128+col%128];unsigned code=(byte>>((col%256>=128)*4))&15;return std::ldexp(double(values[code]),int(s[(uint64_t(col/256)*ng+z/32)*256+col%256])-127);};
  if(mode=="decode"){auto*y=(uint16_t*)out.host;for(int z=0;z<k;++z)for(int col=0;col<n;++col){uint16_t expected=bf(float(weight(col,z)));bad+=expected!=y[uint64_t(z)*n+col];++checked;}}
  else{auto*y=(float*)out.host;refs.resize(uint64_t(g.m)*n);
   for(int col=0;col<n;++col){std::vector<double> weights(k);for(int z=0;z<k;++z)weights[z]=weight(col,z);
    for(int row=0;row<g.m;++row){double ref=b?b[col]:0,absolute=std::abs(ref);for(int z=0;z<k;++z){double product=fp(x[uint64_t(row)*k+z])*weights[z];ref+=product;absolute+=std::abs(product);}double actual=y[uint64_t(row)*n+col],delta=std::abs(actual-ref);refs[uint64_t(row)*n+col]=ref;
     bool ok=std::isfinite(actual)&&delta<=2e-5+2e-6*absolute;bad+=!ok;++checked;err2+=delta*delta;ref2+=ref*ref;maxabs=std::max(maxabs,delta);maxback=std::max(maxback,delta/std::max(absolute,1e-30));
    }
   }save("e"+std::to_string(e)+"_oracle_f64.bin",refs.data(),refs.size()*8);
  }
 }
 double relative=std::sqrt(err2/std::max(ref2,1e-30));
 std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"relative_l2\":%.12g,\"max_abs\":%.12g,\"max_componentwise_backward\":%.12g,\"workspace_bytes\":%llu}\n",checked,bad,relative,maxabs,maxback,(unsigned long long)workspaceBytes);std::fflush(stdout);if(bad||!checked){ck(synDeviceRelease(dev),"failed release");ck(synDestroy(),"failed destroy");return 3;}
 for(int i=0;i<5;++i)run();ck(synStreamSynchronize(stream),"warmup");synEventHandle begin,end;ck(synEventCreate(&begin,dev,EVENT_COLLECT_TIME),"event");ck(synEventCreate(&end,dev,EVENT_COLLECT_TIME),"event");
 int repeats=mode=="tpc"?5:20;if(const char* value=std::getenv("GK_PROBE_REPEATS")){repeats=std::stoi(value);if(repeats<1||repeats>1000)throw std::invalid_argument("repeats out of bounds");}
 for(int sample=0;sample<5;++sample){auto wall=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"event begin");for(int i=0;i<repeats;++i)run();ck(synEventRecord(end,stream),"event end");ck(synEventSynchronize(end),"event sync");uint64_t ns;ck(synEventElapsedTime(&ns,begin,end),"event elapsed");double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-wall).count()/repeats;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"repeats\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,repeats,double(ns)/1000/repeats,us);}
 ck(synStreamSynchronize(stream),"finish");ck(synDeviceRelease(dev),"release");ck(synDestroy(),"destroy");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
