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
int main(int argc,char**argv){if(argc!=2||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;bool overflow_case=std::string(argv[1])=="overflow";unsigned T=3,R=2,E=4,K=64,C=overflow_case?2:3,N=1024;std::vector<int32_t>ids={0,1,0,-1,0,3},inverse(T*R,-1),counts(E,-1),flags(E,-1);std::vector<uint16_t>a(T*R*K),grouped(E*C*K,0xbeef);for(unsigned row=0;row<T*R;++row)for(unsigned k=0;k<K;++k)a[row*K+k]=bf((row+1)*.25f+k/128.f);
 run("gk_mxfp4_group_prepare",{{a.data(),DATA_BF16,1,{K,T*R}},{ids.data(),DATA_I32,2,{R,T}}},{{grouped.data(),DATA_BF16,1,{K,C,E}},{inverse.data(),DATA_I32,2,{R,T}},{counts.data(),DATA_I32,2,{E,1}},{flags.data(),DATA_I32,2,{E,1}}},{int(C),int(E),0});
 unsigned bad=0,checked=0;std::vector<unsigned>seen(E,0);std::vector<uint16_t>wanted(grouped.size(),0);for(unsigned route=0;route<T*R;++route){int e=ids[route];if(e<0)continue;unsigned at=seen[e]++;bad+=inverse[route]!=int(at);++checked;if(at<C)std::memcpy(wanted.data()+(e*C+at)*K,a.data()+route*K,K*2);}for(unsigned e=0;e<E;++e){bad+=counts[e]!=int(seen[e]);bad+=flags[e]!=int(seen[e]>C);checked+=2;}for(unsigned i=0;i<grouped.size();++i){bad+=grouped[i]!=wanted[i];++checked;}
 unsigned tile=512;std::vector<uint8_t>w(E*(N/512)*K*256,0),s(E*(N/512)*(K/32)*512,0);std::vector<uint16_t>lut(512),decoded(tile*K*2,0xbeef);float values[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};for(unsigned i=0;i<256;++i){lut[2*i]=bf(values[i&15]);lut[2*i+1]=bf(values[i>>4]);}
 for(unsigned e=0;e<E;++e)for(unsigned n=0;n<N;++n){unsigned p=(n%512/128)*128+2*(n%64)+(n%128)/64;for(unsigned k=0;k<K;++k){unsigned byte=((e*(N/512)+n/512)*K+k)*256+(p/256)*128+p%128,shift=4*((p%256)/128);w[byte]|=((n+3*k+e)%16)<<shift;s[((e*(N/512)+n/512)*(K/32)+k/32)*512+p]=118+(e+n/16+k/32)%15;}}
 auto stats=run("gk_mxfp4_group_decode",{{w.data(),DATA_U8,0,{256,K,E*(N/512)}},{s.data(),DATA_U8,0,{512,K/32,E*(N/512)}},{lut.data(),DATA_BF16,1,{512,1}},{counts.data(),DATA_I32,2,{E,1}}},{{decoded.data(),DATA_BF16,1,{tile,K,2}}},{int(N/512),2,1,int(C)});
 for(unsigned local=0;local<2;++local)for(unsigned k=0;k<K;++k)for(unsigned n=0;n<tile;++n){unsigned e=local+2,gn=n+512,scale=118+(e+gn/16+k/32)%15;uint16_t expected=counts[e]>0&&counts[e]<=int(C)?bf(std::ldexp(values[(gn+3*k+e)%16],int(scale)-127)):0;bad+=decoded[(local*K+k)*tile+n]!=expected;++checked;}
 std::vector<float>partials(E*C*tile),routing={.125123f,.33123f,.23456f,0,.34567f,.45678f},output(T*tile,-999);for(unsigned e=0;e<E;++e)for(unsigned row=0;row<C;++row)for(unsigned n=0;n<tile;++n)partials[(e*C+row)*tile+n]=float(e*8+row)+float(n)/128;
 run("gk_mxfp4_group_combine",{{partials.data(),DATA_F32,2,{tile,C,E}},{ids.data(),DATA_I32,2,{R,T}},{routing.data(),DATA_F32,2,{R,T}},{inverse.data(),DATA_I32,2,{R,T}},{flags.data(),DATA_I32,2,{E,1}}},{{output.data(),DATA_F32,2,{tile,T}}},{int(C),int(E)});
 for(unsigned t=0;t<T;++t)for(unsigned n=0;n<tile;++n){float wanted=0;bool invalid=false;for(unsigned slot=0;slot<R;++slot){unsigned route=t*R+slot;int e=ids[route];if(e<0)continue;if(flags[e]){invalid=true;break;}wanted=std::fma(partials[(e*C+inverse[route])*tile+n],routing[route],wanted);}bad+=invalid?!std::isnan(output[t*tile+n]):output[t*tile+n]!=wanted;++checked;}
 // Same raw logical values through the new GP epilogue and literal gate.
 std::vector<float>legacy_partials(T*R*tile,0);std::vector<uint16_t>gate(T*R*tile/2,0),legacy_gate(gate.size(),0);
 for(unsigned route=0;route<T*R;++route){int e=ids[route];if(e<0||flags[e])continue;for(unsigned n=0;n<tile;++n){unsigned at=(n/128)*128+(n%2)*64+(n%128)/2;legacy_partials[route*tile+at]=partials[(e*C+inverse[route])*tile+n];}}
 run("gk_mxfp4_group_legacy_gate",{{legacy_partials.data(),DATA_F32,2,{tile*T*R,1}},{ids.data(),DATA_I32,2,{R,T}}},{{legacy_gate.data(),DATA_BF16,1,{tile/2,T*R}}},{1});
 run("gk_mxfp4_group_gate",{{partials.data(),DATA_F32,2,{tile,C,E}},{ids.data(),DATA_I32,2,{R,T}},{routing.data(),DATA_F32,2,{R,T}},{inverse.data(),DATA_I32,2,{R,T}},{flags.data(),DATA_I32,2,{E,1}}},{{gate.data(),DATA_BF16,1,{tile/2,T*R}}},{int(C),int(E)});
 for(unsigned route=0;route<T*R;++route)for(unsigned col=0;col<tile/2;++col){auto got=gate[route*(tile/2)+col];bool invalid=ids[route]>=0&&flags[ids[route]];bad+=invalid?((got&0x7f80)!=0x7f80||!(got&127)):got!=legacy_gate[route*(tile/2)+col];++checked;}
 // Shared-token input gather preserves each BF16 bit for all selected experts.
 std::vector<uint16_t>shared(T*K);for(unsigned t=0;t<T;++t)std::memcpy(shared.data()+t*K,a.data()+t*R*K,K*2);
 run("gk_mxfp4_group_prepare",{{shared.data(),DATA_BF16,1,{K,T}},{ids.data(),DATA_I32,2,{R,T}}},{{grouped.data(),DATA_BF16,1,{K,C,E}},{inverse.data(),DATA_I32,2,{R,T}},{counts.data(),DATA_I32,2,{E,1}},{flags.data(),DATA_I32,2,{E,1}}},{int(C),int(E),1});
 for(unsigned route=0;route<T*R;++route){int e=ids[route];if(e<0||inverse[route]>=int(C))continue;for(unsigned k=0;k<K;++k){bad+=grouped[(e*C+inverse[route])*K+k]!=shared[(route/R)*K+k];++checked;}}
 std::cout<<"{\"case\":\""<<argv[1]<<"\",\"checked\":"<<checked<<",\"bad\":"<<bad<<",\"packed_vector_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_vector_loads\":"<<stats.vectorLoadPerTensor[1]<<",\"device_validated\":false}\n";return bad?3:0;}
