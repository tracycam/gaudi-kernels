#include "../../csrc/ops/mxfp4_exact.hpp"
#include "../../csrc/ops/mxfp4_overhead.hpp"
#include "../../csrc/common/experiment_device.hpp"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static void check(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static std::vector<unsigned char> read_file(const std::string&path){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("open "+path);auto n=f.tellg();f.seekg(0);std::vector<unsigned char>b(n);f.read((char*)b.data(),n);if(!f)throw std::runtime_error("read "+path);return b;}
static void save(const std::string&name,const void*data,size_t bytes){const char*out=std::getenv("PROBE_OUT");if(!out)throw std::runtime_error("PROBE_OUT required");std::ofstream f(std::string(out)+"/"+name,std::ios::binary);f.write((const char*)data,bytes);if(!f)throw std::runtime_error("save "+name);}
static float f32(uint32_t bits){float result;std::memcpy(&result,&bits,4);return result;}
static uint16_t bf(float x){uint32_t bits;std::memcpy(&bits,&x,4);return bits>>16;}
struct Buffer{std::string name;synTensor tensor;synSectionHandle section;uint64_t bytes,address;void*host;bool output;};
int main(int argc,char**argv){synDeviceId device=0;bool acquired=false,initialized=false;try{
 if(argc!=6)throw std::invalid_argument("probe fixture_dir exact|pred|history|prepare splits force_exact timing_repeats");
 std::string fixture=argv[1],mode=argv[2];int splits=std::stoi(argv[3]),force=std::stoi(argv[4]),repeats=std::stoi(argv[5]);
 if(mode!="exact"&&mode!="pred"&&mode!="history"&&mode!="prepare")throw std::invalid_argument("mode");
 int n,k,m,bias,smin,smax;std::ifstream config(fixture+"/config.txt");config>>n>>k>>m>>bias>>smin>>smax;if(!config||n<1||k<1||m<1||repeats<1)throw std::invalid_argument("fixture config");
 if(mode!="exact"&&(m!=1||n%512||k%32))throw std::invalid_argument("aligned M1 required");
 auto geometry=gaudi_kernels::mxfp4_overhead::geometry(n,k,m,splits);splits=geometry.splits;
 auto config_bytes=read_file(fixture+"/fixture.json");save("fixture.json",config_bytes.data(),config_bytes.size());
 std::printf("{\"stage\":\"initialize\",\"module\":%u}\n",gaudi_experiment_module());std::fflush(stdout);
 check(synInitialize(),"initialize");initialized=true;std::printf("{\"stage\":\"acquire_begin\"}\n");std::fflush(stdout);
 check(synDeviceAcquireByModuleId(&device,gaudi_experiment_module()),"assigned module");acquired=true;std::printf("{\"stage\":\"acquired\"}\n");std::fflush(stdout);
 synGraphHandle graph;check(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer> buffers;buffers.reserve(16);
 auto tensor=[&](const std::string&name,synDataType type,std::vector<int>shape,uint64_t bytes,const std::string&file,bool output=false){
  synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=shape.size();for(size_t i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];
  synSectionHandle section;check(synSectionCreate(&section,0,graph),"section");check(synSectionSetPersistent(section,true),"persistent");synTensor t;check(synTensorCreate(&t,&d,section,0),"tensor");
  Buffer b{name,t,section,bytes,0,nullptr,output};check(synHostMalloc(device,bytes,0,&b.host),"host malloc");check(synDeviceMalloc(device,bytes,0,0,&b.address),"device malloc");std::memset(b.host,0,bytes);
  if(!file.empty()){auto raw=read_file(fixture+"/"+file);if(raw.size()!=bytes)throw std::runtime_error("fixture size "+file);std::memcpy(b.host,raw.data(),bytes);}buffers.push_back(b);return t;
 };
 auto activation=tensor("activation",syn_type_bf16,{k,m},uint64_t(k)*m*2,"activation.bf16");size_t activation_index=0;
 synTensor bt=nullptr;if(bias)bt=tensor("bias",syn_type_single,{n,1},uint64_t(n)*4,"bias.u32");
 size_t output_index=0,flags_index=0;
 if(mode=="prepare"){
  output_index=buffers.size();auto copied=tensor("copied",syn_type_bf16,{geometry.kpart,geometry.tasks},uint64_t(geometry.kpart)*geometry.tasks*2,"",true);
  flags_index=buffers.size();auto flags=tensor("flags",syn_type_uint16,{1,geometry.tasks},uint64_t(geometry.tasks)*2,"",true);
  int params[]={n,k,geometry.nblocks,splits,smin,smax,force};synTensor in[]={activation,bt},out[]={copied,flags};
  check(synNodeCreate(graph,in,out,bias?2:1,2,params,sizeof(params),bias?"gk_mxfp4_dispatch_prepare_bias":"gk_mxfp4_dispatch_prepare","prepare_copy_gate",nullptr,nullptr),"prepare node");
 }else{
  uint64_t weight_bytes=uint64_t(n)*((uint64_t(k)+1)/2),scale_bytes=uint64_t(n)*((uint64_t(k)+31)/32);
  auto packed=tensor("packed",mode=="exact"?syn_type_uint8:syn_type_packed_mxfp4,mode=="exact"?std::vector<int>{int(weight_bytes),1}:std::vector<int>{256,n/512*k},weight_bytes,"packed.u8");
  auto scales=tensor("scales",syn_type_uint8,mode=="exact"?std::vector<int>{int(scale_bytes),1}:std::vector<int>{512,n/512*(k/32)},scale_bytes,"scales.u8");
  output_index=buffers.size();auto output=tensor("output",syn_type_single,{n,m},uint64_t(n)*m*4,"",true);
  if(mode=="exact")gaudi_kernels::mxfp4_exact::append(graph,packed,scales,activation,output,bt,n,k,m,gaudi_kernels::mxfp4_exact::Layout::CompactV3);
  else{
   auto lut=tensor("lut",syn_type_bf16,{512,1},1024,"");auto*table=(uint16_t*)buffers.back().host;
   const float q[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};for(int i=0;i<256;++i){table[2*i]=bf(q[i&15]);table[2*i+1]=bf(q[i>>4]);}
   auto mapping=tensor("mapping",syn_type_int32,{geometry.tasks,1},uint64_t(geometry.tasks)*4,"");auto*map=(int32_t*)buffers.back().host;for(int t=0;t<geometry.tasks;++t)map[t]=t;
   if(mode=="pred")gaudi_kernels::mxfp4_exact::append_predispatch_native_m1(graph,packed,scales,activation,lut,mapping,output,bt,n,k,smin,smax,splits,force);
   else gaudi_kernels::mxfp4_overhead::append(graph,packed,scales,activation,lut,mapping,output,bt,geometry,"history");
  }
 }
 std::printf("{\"stage\":\"compile_begin\",\"mode\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%d,\"splits\":%d,\"force_exact\":%d}\n",mode.c_str(),n,k,m,splits,force);std::fflush(stdout);
 synRecipeHandle recipe;check(synGraphCompile(&recipe,graph,"mxfp4_recovered",nullptr),"compile");std::printf("{\"stage\":\"compiled\"}\n");std::fflush(stdout);
 synStreamHandle stream;check(synStreamCreateGeneric(&stream,device,0),"stream");std::vector<synLaunchTensorInfo>launch;
 for(auto&b:buffers){synLaunchTensorInfo t={};t.tensorName=b.name.c_str();t.tensorType=DATA_TENSOR;t.pTensorAddress=b.address;launch.push_back(t);if(!b.output){check(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"copy input");save(b.name+".bin",b.host,b.bytes);}}
 uint64_t workspace_bytes=0,workspace=0;check(synWorkspaceGetSize(&workspace_bytes,recipe),"workspace size");if(workspace_bytes)check(synDeviceMalloc(device,workspace_bytes,0,0,&workspace),"workspace allocation");
 auto run=[&](){check(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};run();
 for(auto&b:buffers)if(b.output)check(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"copy output");check(synStreamSynchronize(stream),"correctness sync");
 size_t checked=0,bad=0,bit_difference=0,nonzero_to_zero=0;double max_error=0,max_backward=0,error2=0,reference2=0;
 if(mode=="prepare"){
  auto*input=(uint16_t*)buffers[activation_index].host;auto*copied=(uint16_t*)buffers[output_index].host;auto*flags=(uint16_t*)buffers[flags_index].host;
  for(int task=0;task<geometry.tasks;++task){for(int z=0;z<geometry.kpart;++z){uint16_t wanted=input[(task%splits)*geometry.kpart+z];bad+=copied[uint64_t(task)*geometry.kpart+z]!=wanted;++checked;}bad+=flags[task]!=unsigned(force!=0);++checked;}
 }else{
  auto expected_raw=read_file(fixture+"/expected.u32"),reference_raw=read_file(fixture+"/reference.f64"),absolute_raw=read_file(fixture+"/sumabs.f64"),flags_raw=read_file(fixture+"/expected_flags.u16");
  if(expected_raw.size()!=uint64_t(n)*m*4||reference_raw.size()!=uint64_t(n)*m*8||absolute_raw.size()!=reference_raw.size())throw std::runtime_error("oracle shape");
  save("expected.u32",expected_raw.data(),expected_raw.size());save("reference.f64",reference_raw.data(),reference_raw.size());save("sumabs.f64",absolute_raw.data(),absolute_raw.size());save("expected_flags.u16",flags_raw.data(),flags_raw.size());
  auto*expected=(const uint32_t*)expected_raw.data();auto*reference=(const double*)reference_raw.data();auto*absolute=(const double*)absolute_raw.data();auto*flags=(const uint16_t*)flags_raw.data();auto*actual=(uint32_t*)buffers[output_index].host;
  for(int row=0;row<m;++row)for(int col=0;col<n;++col){size_t at=uint64_t(row)*n+col;bool exact=mode=="exact"||(mode=="pred"&&(force||flags[row*((n+511)/512)+col/512]));double value=f32(actual[at]),delta=std::abs(value-reference[at]);
   bool okay=exact?actual[at]==expected[at]:(std::isfinite(value)&&delta<=2e-6*absolute[at]+2*double(std::numeric_limits<float>::denorm_min())*(k+1));
   ++checked;bad+=!okay;bit_difference+=actual[at]!=expected[at];nonzero_to_zero+=(expected[at]&0x7fffffff)!=0&&(actual[at]&0x7fffffff)==0;
   if(std::isfinite(delta)){max_error=std::max(max_error,delta);max_backward=std::max(max_backward,delta/std::max(absolute[at],1e-300));error2+=delta*delta;reference2+=reference[at]*reference[at];}
  }
 }
 for(auto&b:buffers)if(b.output)save(b.name+".bin",b.host,b.bytes);
 std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"bit_differences_from_exact\":%zu,\"nonzero_reference_to_zero\":%zu,\"max_abs\":%.12g,\"max_backward\":%.12g,\"relative_l2\":%.12g,\"workspace_bytes\":%llu}\n",checked,bad,bit_difference,nonzero_to_zero,max_error,max_backward,std::sqrt(error2/std::max(reference2,1e-300)),(unsigned long long)workspace_bytes);std::fflush(stdout);
 if(bad){check(synDeviceRelease(device),"failed release");acquired=false;check(synDestroy(),"failed destroy");initialized=false;return 3;}
 int warmups=repeats==1?1:10;for(int i=0;i<warmups;++i)run();check(synStreamSynchronize(stream),"warmup sync");synEventHandle begin,end;check(synEventCreate(&begin,device,EVENT_COLLECT_TIME),"event begin create");check(synEventCreate(&end,device,EVENT_COLLECT_TIME),"event end create");
 for(int sample=0;sample<5;++sample){auto wall=std::chrono::steady_clock::now();check(synEventRecord(begin,stream),"event begin");for(int i=0;i<repeats;++i)run();check(synEventRecord(end,stream),"event end");check(synEventSynchronize(end),"event sync");uint64_t ns=0;check(synEventElapsedTime(&ns,begin,end),"elapsed");double wall_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-wall).count()/repeats;
  std::printf("{\"stage\":\"timing\",\"sample\":%d,\"repeats\":%d,\"warmups\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,repeats,warmups,double(ns)/1000/repeats,wall_us);}
 check(synStreamSynchronize(stream),"final sync");check(synDeviceRelease(device),"release");acquired=false;check(synDestroy(),"destroy");initialized=false;return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());if(acquired)synDeviceRelease(device);if(initialized)synDestroy();return 2;}}
