// Offline actual library/glue/ELFs, complete FP32 output comparisons.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
using Instantiate=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
Tensor tensor(TensorDataType type,unsigned width,unsigned rows){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?width:i==1?rows:1;return t;}
tpc_tests::TensorDesc2 desc(void*data,unsigned width,unsigned rows,unsigned log2){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned n=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=n;d.dimDescriptors[i].stride=stride;stride*=n;}return d;}
template<class T>void write(const char*n,const std::vector<T>&v){std::ofstream f(n,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));}
uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
float bf(uint16_t v){uint32_t u=uint32_t(v)<<16;float f;std::memcpy(&f,&u,4);return f;}
int main(int argc,char**argv){
 if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 const unsigned T=std::atoi(argv[2]),mode=std::atoi(argv[3]),E=8,N=6144,K=256,R=8,L=T*R;if(T<1||T>8||mode>1)return 2;
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h){std::cerr<<dlerror();return 3;}auto fn=(Instantiate)dlsym(h,"InstantiateTpcKernel");if(!fn)return 4;
 std::vector<uint8_t>w(E*N*K/2),s(E*N*K/32);std::vector<uint16_t>a(L*K),lut(512);std::vector<int32_t>ids(L);std::vector<float>old(L*N,-99),candidate(L*N,-99);
 for(unsigned i=0;i<w.size();++i)w[i]=uint8_t(i*47+i/257+i/4093);
 for(unsigned i=0;i<s.size();++i)s[i]=mode?2+(i*7+i/13)%251:118+(i*7+i/13)%15;
 for(unsigned i=0;i<a.size();++i)a[i]=(mode?(i%17==0?0:uint16_t((1+(i*13)%32)*128+(i*37)%128)):uint16_t(0x3c00+(i*37)%1024))^(i%3==0?0x8000:0);
 for(unsigned i=0;i<L;++i)ids[i]=(i*3+1+i/R)%E;
 const uint16_t q[]={0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0,0x8000,0xbf00,0xbf80,0xbfc0,0xc000,0xc040,0xc080,0xc0c0};for(unsigned i=0;i<256;++i){lut[2*i]=q[i&15];lut[2*i+1]=q[i>>4];}
 for(unsigned variant=0;variant<2;++variant){
  auto&y=variant?candidate:old;std::vector<Tensor>inputs={tensor(DATA_U8,256,E*3072),tensor(DATA_U8,512,E*96),tensor(DATA_BF16,K,L),tensor(DATA_BF16,512,1),tensor(DATA_I32,R,T)};Tensor output=tensor(DATA_F32,L*N,1);TensorAccessPattern ia[5]{},oa{};HabanaKernelParams p{};HabanaKernelInstantiation k{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=5;p.outputTensorNr=1;p.inputTensors=inputs.data();p.outputTensors=&output;std::strcpy(p.guid.name,variant?"gk_moe_down_242_scale_tail_experiment":"gk_moe_down_289_control_experiment");std::vector<uint8_t>elf(1<<20);k.kernel.kernelElf=elf.data();k.kernel.elfSize=elf.size();k.inputTensorAccessPattern=ia;k.outputTensorAccessPattern=&oa;if(fn(&p,&k)!=GLUE_SUCCESS)return 5;
  std::vector<tpc_tests::TensorDesc2>d={desc(w.data(),256,E*3072,0),desc(s.data(),512,E*96,0),desc(a.data(),K,L,1),desc(lut.data(),512,1,1),desc(ids.data(),R,T,2),desc(y.data(),L*N,1,2)};VPEStats stats;auto cycles=tpc_tests::RunSimulation(p,k,d,stats);
  std::cout<<"{\"variant\":"<<variant<<",\"cycles_not_device_time\":"<<cycles<<",\"instructions\":"<<stats.instructionsExecuted<<",\"weight_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"activation_loads\":"<<stats.vectorLoadPerTensor[2]<<"}\n";
 }
 unsigned bad=0,finite=0,oraclebad=0;double maxratio=0;
 for(unsigned i=0;i<old.size();++i){bad+=bits(old[i])!=bits(candidate[i]);finite+=std::isfinite(candidate[i]);}
 // Independent original-byte full MAC check on ordinary normal data. The
 // extreme fixture compares exact old/new ISA bits; it is not an FTZ oracle.
 if(!mode)for(unsigned slot=0;slot<L;++slot)for(unsigned n=0;n<N;++n){double exact=0,absolute=0;
  for(unsigned kk=0;kk<K;++kk){unsigned pos=n%512;size_t wi=((ids[slot]*12+n/512)*K+kk)*256+(pos/256)*128+pos%128;unsigned code=(w[wi]>>(4*((pos%256)/128)))&15;unsigned scale=s[((ids[slot]*12+n/512)*8+kk/32)*512+pos];double term=double(bf(a[slot*K+kk]))*std::ldexp(double(bf(q[code])),int(scale)-127);exact+=term;absolute+=std::abs(term);}
  unsigned physical=(n/128)*128+(n%2)*64+(n%128)/2;double error=std::abs(double(candidate[slot*N+physical])-exact);double gamma=(264./16777216)/(1-264./16777216);double limit=gamma*absolute;maxratio=std::max(maxratio,error/(limit+1e-300));oraclebad+=!std::isfinite(candidate[slot*N+physical])||error>limit;
 }
 write("weights.bin",w);write("scales.bin",s);write("activation.bin",a);write("table.bin",lut);write("ids.bin",ids);write("control.bin",old);write("candidate.bin",candidate);
 std::cout<<"{\"tokens\":"<<T<<",\"mode\":"<<mode<<",\"words\":"<<old.size()<<",\"bit_mismatches\":"<<bad<<",\"finite_words\":"<<finite<<",\"ordinary_FP32_bound_failures\":"<<oraclebad<<",\"max_bound_ratio\":"<<maxratio<<",\"device_tested\":false}\n";return bad||oraclebad?6:0;
}
