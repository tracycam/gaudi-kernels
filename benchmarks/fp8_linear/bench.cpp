#include "ops/fp8_linear.hpp"
#include "common/experiment_device.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>
static void ck(synStatus s,const char*w){if(s!=synSuccess){std::fprintf(stderr,"FAIL %s status=%d\n",w,int(s));std::exit(2);}}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static void read(const std::string&p,void*v,size_t n){FILE*f=std::fopen(p.c_str(),"rb");if(!f||std::fread(v,1,n,f)!=n){std::fprintf(stderr,"read failed %s\n",p.c_str());std::exit(4);}std::fclose(f);}
struct Buffer{synTensor t;std::string name;uint64_t bytes,address;void*host;};
static int probe(synDeviceId device,const std::string& mode,int m,int n,int k,const std::string& fixture){
 bool a8=mode=="w8a8";if(m<1||n<1||k<1||(!a8&&mode!="w8a16"))return 2;
 std::string quantizer=std::getenv("GK_FP8_QUANTIZER")?std::getenv("GK_FP8_QUANTIZER"):"baseline";
 bool use_lut=quantizer=="lut_single"||quantizer=="lut_single4";
 synGraphHandle graph;ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");std::vector<Buffer>bufs;bufs.reserve(6);
 auto tensor=[&](const char*name,synDataType dtype,int d0,int d1,uint64_t bytes){
  synSectionHandle sec;ck(synSectionCreate(&sec,0,graph),"section");ck(synSectionSetPersistent(sec,true),"persistent");
  synTensorDescriptor d={};d.m_name=name;d.m_dataType=dtype;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;
  Buffer b={nullptr,name,bytes,0,nullptr};ck(synTensorCreate(&b.t,&d,sec,0),"tensor");
  if(dtype==syn_type_fp8_143){synFpQuantParam p{1.,7};synFpQuantMetadata meta{dtype,&p,1};ck(synTensorSetQuantizationData(b.t,SYN_FP_QUANT_METADATA,&meta,sizeof(meta)),"fp8 metadata");}
  ck(synHostMalloc(device,bytes,0,&b.host),"host allocation");ck(synDeviceMalloc(device,bytes,0,0,&b.address),"device allocation");bufs.push_back(b);return b.t;
 };
 bool weight_kn=std::getenv("GK_FP8_WEIGHT_KN")&&std::string(std::getenv("GK_FP8_WEIGHT_KN"))=="1";
 synTensor w=tensor("prepared_weight",syn_type_fp8_143,weight_kn?n:k,weight_kn?k:n,uint64_t(n)*k);
 synTensor x=tensor("activation",syn_type_bf16,k,m,uint64_t(m)*k*2);
 synTensor ws=tensor("prepared_channel_scales",syn_type_single,n,1,n*4);
 synTensor bias=tensor("bias",syn_type_single,n,1,n*4);
 synTensor table=use_lut?tensor("reciprocal_table",syn_type_single,129,1,129*4):nullptr;
 synTensor y=tensor("output",syn_type_bf16,n,m,uint64_t(m)*n*2);
 const char*files[]={"weight_native.bin","activation.bin","scales_prepared.bin","bias.bin"};
 for(int i=0;i<4;++i)read(fixture+"/"+files[i],bufs[i].host,bufs[i].bytes);
 if(use_lut){const char* path=std::getenv("GK_FP8_RECIPROCAL_TABLE");if(!path)return 2;read(path,bufs[4].host,bufs[4].bytes);}
 // Verify the reusable host preparer against the saved immutable original.
 std::vector<uint8_t>original(uint64_t(n)*k),prepared(original.size());read(fixture+"/weight_original.bin",original.data(),original.size());
 std::vector<float>original_scales(n),prepared_scales(n);read(fixture+"/scales_original.bin",original_scales.data(),n*4);
 if(!gaudi_kernels::prepare_fp8_linear_channels(original.data(),prepared.data(),original_scales.data(),prepared_scales.data(),n,k)||std::memcmp(prepared.data(),bufs[0].host,prepared.size())||std::memcmp(prepared_scales.data(),bufs[2].host,n*4))return 5;
 if(weight_kn)gaudi_kernels::transpose_prepared_fp8_linear(prepared.data(),static_cast<uint8_t*>(bufs[0].host),n,k);
 uint64_t budget=0;synDeviceAttribute attr=DEVICE_ATTRIBUTE_MAX_RMW_SIZE;ck(synDeviceGetAttribute(&budget,&attr,1,device),"rmw capacity");
 if(const char*rmw=std::getenv("GK_FP8_RMW"))if(std::string(rmw)=="0")budget=0;
 gaudi_kernels::FP8LinearSpec spec{uint32_t(m),uint32_t(n),uint32_t(k),a8?gaudi_kernels::FP8Activation::PerTokenE4M3:gaudi_kernels::FP8Activation::BF16,budget};
 spec.weight_n_contiguous=weight_kn;
 spec.reciprocal_table=table;
 if(quantizer=="lut_single")spec.experimental_quantizer=gaudi_kernels::FP8ExperimentalQuantizer::LutSingle;
 else if(quantizer=="lut_single4")spec.experimental_quantizer=gaudi_kernels::FP8ExperimentalQuantizer::LutFour;
 else if(quantizer=="amax16")spec.experimental_quantizer=gaudi_kernels::FP8ExperimentalQuantizer::IntegerAmax;
 else if(quantizer!="baseline")return 2;
 if(const char*rows=std::getenv("GK_FP8_DECODE_ROWS"))spec.decode_row_block=std::atoi(rows);
 if(const char*force=std::getenv("GK_FP8_FORCE_SRAM"))spec.force_intermediate_sram=std::string(force)=="1";
 if(const char*fast=std::getenv("GK_FP8_QUANT_FAST"))spec.native_quant_conversion=std::string(fast)=="1";
 if(const char*fast=std::getenv("GK_FP8_EPILOGUE_FAST"))spec.interleaved_epilogue=std::string(fast)=="1";
 if(const char*fast=std::getenv("GK_FP8_EPILOGUE_ROWS"))spec.reused_scale_epilogue=std::string(fast)=="1";
 gaudi_kernels::FP8LinearBuild build;ck(gaudi_kernels::add_fp8_linear(graph,spec,x,w,ws,bias,y,&build),"add reusable linear");
 synRecipeHandle recipe;ck(synGraphCompile(&recipe,graph,"fp8_linear",nullptr),"compile");
 synStreamHandle stream;ck(synStreamCreateGeneric(&stream,device,0),"stream");std::vector<synLaunchTensorInfo>launch(bufs.size());
 for(size_t i=0;i<bufs.size();++i){auto&b=bufs[i];launch[i]={};launch[i].tensorName=b.name.c_str();launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=b.address;if(i+1<bufs.size())ck(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"input copy");}
 uint64_t workbytes=0,workspace=0;ck(synWorkspaceGetSize(&workbytes,recipe),"workspace size");if(workbytes)ck(synDeviceMalloc(device,workbytes,0,0,&workspace),"workspace");
 auto run=[&](){ck(synLaunch(stream,launch.data(),launch.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");};
 run();auto&out=bufs.back();ck(synMemCopyAsync(stream,out.address,out.bytes,uint64_t(out.host),DRAM_TO_HOST),"output copy");ck(synStreamSynchronize(stream),"check sync");
 auto*f=std::fopen("output.bin","wb");if(!f||std::fwrite(out.host,1,out.bytes,f)!=out.bytes)return 4;std::fclose(f);
 size_t count=uint64_t(m)*n;std::vector<double>ref(count),contract(count),sumabs(count),gpu(count);read(fixture+"/gpu_style_w8a8_fp64.bin",gpu.data(),count*8);read(fixture+"/reference_fp64.bin",ref.data(),count*8);read(fixture+(a8?"/adapted_w8a8_fp64.bin":"/adapted_w8a16_fp64.bin"),contract.data(),count*8);read(fixture+"/sumabs_fp64.bin",sumabs.data(),count*8);
 double err=0,energy=0,ce=0,cn=0,maxabs=0,maxnormalized=0,gpuerror=0;size_t bad=0;
 for(size_t i=0;i<count;++i){double value=fp(static_cast<uint16_t*>(out.host)[i]);double e=value-ref[i],c=value-contract[i];float gv=gpu[i];uint32_t bits;std::memcpy(&bits,&gv,4);bits+=0x7fff+((bits>>16)&1);double gr=fp(bits>>16);gpuerror+=(gr-ref[i])*(gr-ref[i]);bad+=!std::isfinite(value);err+=e*e;energy+=ref[i]*ref[i];ce+=c*c;cn+=contract[i]*contract[i];maxabs=std::max(maxabs,std::abs(e));maxnormalized=std::max(maxnormalized,std::abs(c)/std::max(sumabs[i],1e-20));}
 double relative=std::sqrt(err/std::max(energy,1e-30)),cr=std::sqrt(ce/std::max(cn,1e-30));
 bool pass=bad==0&&maxnormalized<0.0041&&(cr<0.0041||std::sqrt(ce/count)<1e-8)&&err<=1.21*gpuerror+count*1e-12;
 std::printf("{\"stage\":\"correctness\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"checked\":%zu,\"bad\":%zu,\"pass\":%s,\"relative_fp64\":%.12g,\"relative_adapted_contract\":%.12g,\"max_absolute_fp64\":%.12g,\"max_error_over_sumabs\":%.12g,\"workspace_bytes\":%llu,\"rmw_bytes\":%llu}\n",mode.c_str(),m,n,k,count,bad,pass?"true":"false",relative,cr,maxabs,maxnormalized,(unsigned long long)workbytes,(unsigned long long)build.rmw_bytes);std::fflush(stdout);if(!pass)return 3;
 for(int i=0;i<10;++i)run();ck(synStreamSynchronize(stream),"warmup");
 synEventHandle begin,end;ck(synEventCreate(&begin,device,EVENT_COLLECT_TIME),"begin");ck(synEventCreate(&end,device,EVENT_COLLECT_TIME),"end");
 for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();ck(synEventRecord(begin,stream),"record begin");for(int i=0;i<50;++i)run();ck(synEventRecord(end,stream),"record end");ck(synEventSynchronize(end),"event sync");uint64_t ns=0;ck(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/50.;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,ns/50000.,wall);}
 ck(synStreamSynchronize(stream),"finish");return 0;
}

int main(int argc,char**argv){
 if(argc!=6&&argc!=3)return 2;
 ck(synInitialize(),"initialize");synDeviceId device;ck(synDeviceAcquireByModuleId(&device,gaudi_experiment_module()),"explicit module acquire");
 int result=0;
 if(argc==3&&std::string(argv[1])=="batch"){
  std::ifstream file(argv[2]);if(!file)return 4;std::string line;
  while(std::getline(file,line)){
   std::istringstream row(line);std::string mode,fixture,output,variant;int m,n,k;
   if(!(row>>mode>>m>>n>>k>>fixture>>output>>variant))return 4;
   if(chdir(output.c_str())||!std::freopen("run.log","w",stdout))return 4;
   setenv("GK_FP8_QUANTIZER",variant.c_str(),1);
   result=probe(device,mode,m,n,k,fixture);std::fflush(stdout);if(result)break;
  }
 }else if(argc==6)result=probe(device,argv[1],std::atoi(argv[2]),std::atoi(argv[3]),std::atoi(argv[4]),argv[5]);
 else result=2;
 ck(synDeviceRelease(device),"release");ck(synDestroy(),"destroy");return result;
}
