#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
struct View {void*data;TensorDataType type;unsigned log2;std::vector<unsigned>shape;};
static uint16_t bf(float x){uint32_t u;std::memcpy(&u,&x,4);return u>>16;}
static VPEStats run(const char*guid,const std::vector<View>&inputs,const std::vector<View>&outputs,std::vector<int>params){
 std::vector<Tensor>in,out;std::vector<tpc_tests::TensorDesc2>desc;
 auto add=[&](const View&v,std::vector<Tensor>&target){Tensor t{};t.geometry.dims=v.shape.size();t.geometry.dataType=v.type;t.quantizationParam.scale=1;tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=uint64_t(v.data);d.configuration=v.log2|(((1u<<v.shape.size())-1)<<8)|((v.shape.size()-1)<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i<v.shape.size()?v.shape[i]:1;t.geometry.maxSizes[i]=t.geometry.minSizes[i]=size;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}target.push_back(t);desc.push_back(d);};for(auto&v:inputs)add(v,in);for(auto&v:outputs)add(v,out);
 TensorAccessPattern ia[8]{},oa[4]{};AuxTensor aux[16]{};std::vector<uint8_t>elf(1<<20);HabanaKernelParams p{};HabanaKernelInstantiation q{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=in.size();p.outputTensorNr=out.size();p.inputTensors=in.data();p.outputTensors=out.data();std::strcpy(p.guid.name,guid);p.nodeParams.nodeParams=params.data();p.nodeParams.nodeParamsSize=params.size()*4;q.inputTensorAccessPattern=ia;q.outputTensorAccessPattern=oa;q.auxiliaryTensors=aux;q.kernel.kernelElf=elf.data();q.kernel.elfSize=elf.size();if(InstantiateTpcKernel(&p,&q)!=GLUE_SUCCESS)throw 1;VPEStats stats;tpc_tests::RunSimulation(p,q,desc,stats);return stats;
}

static float f32(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
struct Bucket {unsigned lo,cap,slots,first,row;};
int main(int argc,char**argv){
 if(argc!=2||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::string kind=argv[1];unsigned T=kind=="boundaries"?257:8,R=kind=="boundaries"?4:2,E=kind=="boundaries"?12:kind=="violation"?4:384,K=32,N=512;
 std::vector<int32_t>ids(T*R,-1);
 if(kind=="boundaries"){unsigned cursor=0,e=0;for(unsigned count:{1,16,17,32,33,64,65,128,129,256,257}){for(unsigned j=0;j<count;++j){unsigned at=cursor++;ids[(at%T)*R+at/T]=e;}++e;}}
 else if(kind=="violation")std::fill(ids.begin(),ids.end(),0);
 else for(unsigned t=0;t<T;++t)for(unsigned r=0;r<R;++r)ids[t*R+r]=(t+r)%3==0?-1:int((t*R+r)%6);
 if(kind=="ragged")ids.back()=E+2;
 std::vector<int32_t>counts(E,-1),status(E,-1);std::vector<unsigned>want_counts(E,0);
 for(auto e:ids)if(e>=0&&unsigned(e)<E)++want_counts[e];
 run("gk_mxfp4_bucket_count",{{ids.data(),DATA_I32,2,{R,T}}},{{counts.data(),DATA_I32,2,{E,1}},{status.data(),DATA_I32,2,{E,1}}},{});
 unsigned bad=0,checked=0;for(unsigned e=0;e<E;++e){bad+=counts[e]!=int(want_counts[e]);bad+=status[e]!=int(want_counts[e]>T);checked+=2;}
 unsigned slots=0,rows=0;std::vector<Bucket>buckets;
 for(unsigned lo=1,hi=16;lo<=T;lo=hi+1,hi*=2){unsigned cap=std::min(T,hi),size=std::min(E,T*R/lo);buckets.push_back({lo,cap,size,slots,rows});slots+=size;rows+=size*cap;}
 std::vector<uint16_t>a(T*K);for(unsigned t=0;t<T;++t)for(unsigned k=0;k<K;++k)a[t*K+k]=bf((int(t%5)-2)*.25f+k/128.f);
 std::vector<uint16_t>grouped(rows*K,0xbeef),want_grouped(rows*K,0);std::vector<int32_t>inverse(T*R,-1),map(slots,-99),want_map(slots,-1),want_inverse(T*R,-1);std::vector<unsigned>row_counts(slots,0),row_base(slots,0);
 for(auto b:buckets){unsigned at=0;for(unsigned e=0;e<E;++e)if(want_counts[e]>=b.lo&&want_counts[e]<=b.cap){if(at>=b.slots)throw 10;unsigned slot=b.first+at,first=b.row+at*b.cap;++at;want_map[slot]=e;row_counts[slot]=want_counts[e];row_base[slot]=first;unsigned row=0;for(unsigned route=0;route<T*R;++route)if(ids[route]==int(e)){want_inverse[route]=first+row;std::memcpy(want_grouped.data()+(first+row)*K,a.data()+(route/R)*K,K*2);++row;}}}
 run("gk_mxfp4_bucket_gather",{{a.data(),DATA_BF16,1,{K,T}},{ids.data(),DATA_I32,2,{R,T}},{counts.data(),DATA_I32,2,{E,1}}},{{grouped.data(),DATA_BF16,1,{K,rows}},{inverse.data(),DATA_I32,2,{R,T}},{map.data(),DATA_I32,2,{slots,1}}},{int(E),1});
 for(unsigned i=0;i<grouped.size();++i){bad+=grouped[i]!=want_grouped[i];++checked;}for(unsigned i=0;i<slots;++i){bad+=map[i]!=want_map[i];++checked;}for(unsigned i=0;i<ids.size();++i)if(want_inverse[i]>=0){bad+=inverse[i]!=want_inverse[i];++checked;}
 std::vector<uint8_t>w(E*K*256,0),s(E*(K/32)*512,0);std::vector<uint16_t>lut(512),decoded(slots*K*N,0xbeef);float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
 for(unsigned i=0;i<256;++i){lut[2*i]=bf(values[i&15]);lut[2*i+1]=bf(values[i>>4]);}
 for(unsigned e=0;e<E;++e)for(unsigned n=0;n<N;++n){unsigned p=(n/128)*128+2*(n%64)+(n%128)/64;for(unsigned k=0;k<K;++k){unsigned byte=(e*K+k)*256+(p/256)*128+p%128,shift=4*((p%256)/128);w[byte]|=((n+3*k+e)%16)<<shift;s[(e*(K/32)+k/32)*512+p]=124+(e+n/16+k/32)%4;}}
 auto stats=run("gk_mxfp4_bucket_decode",{{w.data(),DATA_U8,0,{256,K,E}},{s.data(),DATA_U8,0,{512,K/32,E}},{lut.data(),DATA_BF16,1,{512,1}},{map.data(),DATA_I32,2,{slots,1}}},{{decoded.data(),DATA_BF16,1,{N,K,slots}}},{1,0,0});
 unsigned active=0;for(unsigned slot=0;slot<slots;++slot){int e=want_map[slot];active+=e>=0;for(unsigned k=0;k<K;++k)for(unsigned n=0;n<N;++n){uint16_t expected=e<0?0:bf(std::ldexp(values[(n+3*k+e)%16],int(124+(e+n/16+k/32)%4)-127));bad+=decoded[(slot*K+k)*N+n]!=expected;++checked;}}
 bad+=stats.vectorLoadPerTensor[0]!=active*K*2;bad+=stats.vectorLoadPerTensor[1]!=active*(K/32)*4;checked+=2;
 // CPU MAC emulates only the missing MME engine; TPC count/gather/decode and
 // final combine/GP are actual compiled ISA. Native recipe validation is separate.
 std::vector<float>partial(rows*N,0),routing(T*R),combined(T*N,-999);for(unsigned i=0;i<T*R;++i)routing[i]=.125f+float(i%3)/16;
 for(unsigned slot=0;slot<slots;++slot)if(want_map[slot]>=0)for(unsigned row=0;row<row_counts[slot];++row)for(unsigned n=0;n<N;++n){float sum=0;for(unsigned k=0;k<K;++k)sum=std::fma(f32(grouped[(row_base[slot]+row)*K+k]),f32(decoded[(slot*K+k)*N+n]),sum);partial[(row_base[slot]+row)*N+n]=sum;}
 run("gk_mxfp4_bucket_combine",{{partial.data(),DATA_F32,2,{N,rows}},{ids.data(),DATA_I32,2,{R,T}},{routing.data(),DATA_F32,2,{R,T}},{inverse.data(),DATA_I32,2,{R,T}},{status.data(),DATA_I32,2,{E,1}}},{{combined.data(),DATA_F32,2,{N,T}}},{int(E)});
 for(unsigned t=0;t<T;++t)for(unsigned n=0;n<N;++n){float expected=0;bool invalid=false;for(unsigned r=0;r<R;++r){unsigned route=t*R+r;int e=ids[route];if(e<0||unsigned(e)>=E)continue;if(status[e]){invalid=true;break;}expected=std::fma(partial[want_inverse[route]*N+n],routing[route],expected);}bad+=invalid?!std::isnan(combined[t*N+n]):combined[t*N+n]!=expected;++checked;}
 std::vector<float>legacy(T*R*N,0);std::vector<uint16_t>gate(T*R*N/2),reference(gate.size());
 for(unsigned route=0;route<T*R;++route)if(want_inverse[route]>=0)for(unsigned n=0;n<N;++n){unsigned physical=(n/128)*128+(n%2)*64+(n%128)/2;legacy[route*N+physical]=partial[want_inverse[route]*N+n];}
 run("gk_mxfp4_bucket_legacy_gate",{{legacy.data(),DATA_F32,2,{N*T*R,1}},{ids.data(),DATA_I32,2,{R,T}}},{{reference.data(),DATA_BF16,1,{N/2,T*R}}},{1});
 run("gk_mxfp4_bucket_gate",{{partial.data(),DATA_F32,2,{N,rows}},{ids.data(),DATA_I32,2,{R,T}},{routing.data(),DATA_F32,2,{R,T}},{inverse.data(),DATA_I32,2,{R,T}},{status.data(),DATA_I32,2,{E,1}}},{{gate.data(),DATA_BF16,1,{N/2,T*R}}},{int(E)});
 for(unsigned route=0;route<T*R;++route)for(unsigned n=0;n<N/2;++n){int e=ids[route];bool invalid=e>=0&&unsigned(e)<E&&status[e];uint16_t got=gate[route*(N/2)+n];bad+=invalid?((got&0x7f80)!=0x7f80||!(got&127)):got!=reference[route*(N/2)+n];++checked;}
 std::cout<<"{\"case\":\""<<kind<<"\",\"checked\":"<<checked<<",\"bad\":"<<bad<<",\"T\":"<<T<<",\"E\":"<<E<<",\"row_slots\":"<<rows<<",\"expert_slots\":"<<slots<<",\"active_experts\":"<<active<<",\"packed_vector_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_vector_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"device_validated\":false,\"MME_emulated_on_CPU\":true}\n";
 return bad?3:0;
}
