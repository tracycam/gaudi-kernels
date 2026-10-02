// CPU instruction simulator, original deployed down vs experimental fused M1.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
Tensor tensor(TensorDataType type,unsigned width,unsigned rows){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;t.quantizationParam.scale=1;for(unsigned i=0;i<5;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=i==0?width:i==1?rows:1;return t;}
tpc_tests::TensorDesc2 desc(void* data,unsigned width,unsigned rows,unsigned log2){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=(uint64_t)data;d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned n=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=n;d.dimDescriptors[i].stride=stride;stride*=n;}return d;}
template<class T>void write(const char*name,const std::vector<T>&v){std::ofstream f(name,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));}
float bf16(uint16_t v){uint32_t bits=uint32_t(v)<<16;float f;std::memcpy(&f,&bits,4);return f;}
uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
int main(int argc,char**argv){
 if(argc!=4||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 constexpr unsigned E=8,N=6144,K=256,R=8;int mode=std::atoi(argv[3]);
 std::vector<uint8_t>w(E*N*K/2),s(E*N*K/32),directions(512);std::vector<uint16_t>a(R*K),lut(512);std::vector<int32_t>ids(R);std::vector<float>routing(R);
 for(unsigned i=0;i<w.size();++i)w[i]=uint8_t(i*47+i/257+i/4093);
 for(unsigned i=0;i<s.size();++i)s[i]=mode?118+(i*7+i/13)%15:123+(i*7+i/13)%7;
 for(unsigned i=0;i<a.size();++i)a[i]=uint16_t(0x3c00+(i*37)%1024)^(i%3==0?0x8000:0);
 for(unsigned i=0;i<R;++i){ids[i]=(i*3+1)%E;routing[i]=mode?std::ldexp(float(int(i)-3)+.125f,int(i)-4):float(i+1)/32.f+std::ldexp(1.f,-18);}
 const uint16_t qbits[]={0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0,0x8000,0xbf00,0xbf80,0xbfc0,0xc000,0xc040,0xc080,0xc0c0};
 for(unsigned i=0;i<256;++i){lut[2*i]=qbits[i&15];lut[2*i+1]=qbits[i>>4];}
 for(unsigned parity=0;parity<2;++parity)for(unsigned byte=0;byte<256;++byte){unsigned lane=byte/4;directions[parity*256+byte]=(lane%16)/2+((lane/16)%2)*32+(lane%2==parity?128:0);}
 std::vector<float>partial(R*N,-99),candidate(N,-99),reference(N),oracle(N);std::vector<unsigned>cycles;
 for(unsigned variant=0;variant<2;++variant){
  bool narrow=variant;auto& y=narrow?candidate:partial;unsigned wf=narrow?128:256,sf=narrow?256:512;
  std::ifstream f(argv[1+variant],std::ios::binary|std::ios::ate);if(!f)return 3;std::vector<uint8_t>elf(f.tellg());f.seekg(0);f.read((char*)elf.data(),elf.size());
  std::vector<Tensor>inputs={tensor(DATA_U8,wf,w.size()/wf),tensor(DATA_U8,sf,s.size()/sf),tensor(DATA_BF16,K,R),tensor(DATA_BF16,512,1),tensor(DATA_I32,R,1)};
  std::vector<tpc_tests::TensorDesc2>d={desc(w.data(),wf,w.size()/wf,0),desc(s.data(),sf,s.size()/sf,0),desc(a.data(),K,R,1),desc(lut.data(),512,1,1),desc(ids.data(),R,1,2)};
  if(narrow){inputs.push_back(tensor(DATA_F32,R,1));inputs.push_back(tensor(DATA_U8,256,2));d.push_back(desc(routing.data(),R,1,2));d.push_back(desc(directions.data(),256,2,0));}
  Tensor output=tensor(DATA_F32,y.size(),1);d.push_back(desc(y.data(),y.size(),1,2));std::vector<TensorAccessPattern>ia(inputs.size());TensorAccessPattern oa{};for(auto& x:ia)x.allRequired=1;oa.allRequired=1;
  HabanaKernelParams in{};HabanaKernelInstantiation out{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=&output;std::strcpy(in.guid.name,narrow?"down_combine_functional_scaffold":"deployed_down_control");out.indexSpaceRank=1;out.indexSpaceGeometry[0]=narrow?24:96;out.inputTensorAccessPattern=ia.data();out.outputTensorAccessPattern=&oa;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();out.kernel.paramsNr=narrow?0:1;out.kernel.scalarParams[0]=(uint64_t(1)<<32)/R;
  VPEStats stats;cycles.push_back(tpc_tests::RunSimulation(in,out,d,stats));
  std::cout<<"{\"variant\":"<<variant<<",\"sim_cycles\":"<<cycles.back()<<",\"weight_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"activation_loads\":"<<stats.vectorLoadPerTensor[2]<<"}\n";
 }
 unsigned bad=0,oracle_bad=0;double error2=0,ref2=0,maxabs=0,maxback=0;
 for(unsigned n=0;n<N;++n){unsigned internal=(n/128)*128+(n%128)/2+(n%2)*64;float sum=0;double exact=0,sumabs=0;
  for(unsigned route=0;route<R;++route){sum=std::fma(partial[route*N+internal],routing[route],sum);double dot=0,absolute=0;
   for(unsigned k=0;k<K;++k){unsigned pos=n%512;size_t wi=((ids[route]*12+n/512)*K+k)*256+(pos/256)*128+pos%128;unsigned code=(w[wi]>>(4*((pos%256)/128)))&15;unsigned scale=s[((ids[route]*12+n/512)*8+k/32)*512+pos];double term=bf16(a[route*K+k])*std::ldexp(double(bf16(qbits[code])),int(scale)-127);dot+=term;absolute+=std::abs(term);}
   exact+=dot*routing[route];sumabs+=absolute*std::abs(routing[route]);
  }
  // precision_fix combines even/odd via add of shuffled value and +0.
  reference[n]=sum+0.f;oracle[n]=exact;double error=double(candidate[n])-exact;maxabs=std::max(maxabs,std::abs(error));maxback=std::max(maxback,std::abs(error)/(sumabs+1e-300));error2+=error*error;ref2+=exact*exact;
  if(!std::isfinite(candidate[n])||std::abs(error)>2e-6*sumabs+1e-35)++oracle_bad;
  if(bits(reference[n])!=bits(candidate[n])){if(bad<8)std::cerr<<"lane "<<n<<" old="<<reference[n]<<" candidate="<<candidate[n]<<" bits="<<std::hex<<bits(reference[n])<<"/"<<bits(candidate[n])<<std::dec<<"\n";++bad;}
 }
 write("weights.bin",w);write("scales.bin",s);write("activation.bin",a);write("routing.bin",routing);write("ids.bin",ids);write("directions.bin",directions);write("table.bin",lut);write("old-partial.bin",partial);write("expected-combine.bin",reference);write("candidate.bin",candidate);write("fp64-rounded-oracle.bin",oracle);
 std::cout<<"{\"mode\":"<<mode<<",\"checked\":"<<N<<",\"bit_mismatches\":"<<bad<<",\"fp64_gate_failures\":"<<oracle_bad<<",\"max_absolute\":"<<maxabs<<",\"relative_l2\":"<<std::sqrt(error2/ref2)<<",\"max_sumabs_backward_error\":"<<maxback<<",\"device_used\":false}\n";return bad||oracle_bad?4:0;
}
