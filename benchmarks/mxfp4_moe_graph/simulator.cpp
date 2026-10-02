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

struct Bucket{unsigned lo,cap,slots,first,row;};
int main(int argc,char**argv){
 if(argc!=2||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 std::string kind=argv[1];unsigned T=kind=="boundary"?33:kind=="small"?2:8,R=2,E=6,K=128,N=512;bool bucket=kind!="capt";
 std::vector<int>ids(T*R,-1);for(unsigned t=0;t<T;++t){ids[t*R]=0;ids[t*R+1]=t<17?1:t<32?2:3;}
 if(kind=="ragged"){ids[1]=-1;ids[4]=E;ids[7]=-5;}if(kind=="overflow")std::fill(ids.begin(),ids.end(),0);
 std::vector<int>counts(E,-1),status(E,-1),wantcounts(E,0);for(int e:ids)if(e>=0&&e<int(E))++wantcounts[e];
 unsigned bad=0,checked=0;auto eq=[&](auto a,auto b){bad+=a!=b;++checked;};
 run("gk_mxfp4_graph_count",{{ids.data(),DATA_I32,2,{R,T}}},{{counts.data(),DATA_I32,2,{E,1}},{status.data(),DATA_I32,2,{E,1}}},{});
 for(unsigned e=0;e<E;++e){eq(counts[e],wantcounts[e]);eq(status[e],int(wantcounts[e]>int(T)));}
 unsigned slots=0,rows=0;std::vector<Bucket>bs;
 if(!bucket){bs.push_back({1,T,E,0,0});slots=E;rows=E*T;}
 else for(unsigned lo=1,hi=16;lo<=T;lo=hi+1,hi*=2){unsigned cap=std::min(T,hi),count=std::min(E,T*R/lo);bs.push_back({lo,cap,count,slots,rows});slots+=count;rows+=cap*count;}
 std::vector<int>map(slots,-99),wantmap(slots,-1),inv(T*R,-99),wantinv(T*R,-99);std::vector<unsigned>base(slots),caps(slots);
 for(auto b:bs){unsigned at=0;for(unsigned i=0;i<b.slots;++i){base[b.first+i]=b.row+i*b.cap;caps[b.first+i]=b.cap;}for(unsigned e=0;e<E;++e)if(wantcounts[e]>=int(b.lo)&&wantcounts[e]<=int(b.cap)){unsigned slot=b.first+(bucket?at++:e);wantmap[slot]=e;unsigned row=base[slot];for(unsigned route=0;route<T*R;++route)if(ids[route]==int(e))wantinv[route]=row++;}}
 run("gk_mxfp4_graph_plan",{{ids.data(),DATA_I32,2,{R,T}},{counts.data(),DATA_I32,2,{E,1}}},{{inv.data(),DATA_I32,2,{R,T}},{map.data(),DATA_I32,2,{slots,1}}},{int(E),int(bucket)});
 for(unsigned i=0;i<slots;++i)eq(map[i],wantmap[i]);for(unsigned i=0;i<T*R;++i)eq(inv[i],wantinv[i]);
 std::vector<uint16_t>a(T*K);for(unsigned i=0;i<a.size();++i)a[i]=bf((int(i%31)-15)*.03125f);
 // Fragment boundaries include partial last expert batch. Compare every zero pad.
 for(auto b:bs)for(unsigned start=0;start<b.slots;start+=2){unsigned batch=std::min(2u,b.slots-start),slot=b.first+start;
  std::vector<uint16_t>got(batch*b.cap*K,0xbeef),want(got.size(),0);
  for(unsigned local=0;local<batch;++local){int e=wantmap[slot+local];unsigned row=0;if(e>=0)for(unsigned route=0;route<T*R;++route)if(ids[route]==e){std::memcpy(want.data()+(local*b.cap+row++)*K,a.data()+(route/R)*K,K*2);}}
  run("gk_mxfp4_graph_gather",{{a.data(),DATA_BF16,1,{K,T}},{ids.data(),DATA_I32,2,{R,T}},{counts.data(),DATA_I32,2,{E,1}},{map.data(),DATA_I32,2,{slots,1}}},{{got.data(),DATA_BF16,1,{K,b.cap,batch}}},{int(b.cap),int(slot)});
  for(unsigned i=0;i<got.size();++i)eq(got[i],want[i]);
  std::vector<float>partial(batch*b.cap*N),legacy(partial.size());std::vector<int>dummy(batch*b.cap,0);
  for(unsigned local=0;local<batch;++local)for(unsigned row=0;row<b.cap;++row)for(unsigned n=0;n<N;++n){unsigned index=(local*b.cap+row)*N+n;float v=std::ldexp(float(int((row*29+n*13+local)%127)-63),-5);if(wantmap[slot+local]<0||row>=unsigned(wantcounts[wantmap[slot+local]]))v=NAN;partial[index]=v;unsigned physical=(n/128)*128+(n%2)*64+(n%128)/2;legacy[(local*b.cap+row)*N+physical]=v;}
  std::vector<uint16_t>gate(batch*b.cap*256,0xbeef),reference(gate.size(),0xbeef);
  run("gk_mxfp4_graph_gate_rows",{{partial.data(),DATA_F32,2,{N,b.cap,batch}},{map.data(),DATA_I32,2,{slots,1}},{counts.data(),DATA_I32,2,{E,1}}},{{gate.data(),DATA_BF16,1,{256,b.cap,batch}}},{int(slot)});
  run("gk_mxfp4_graph_literal_gate",{{legacy.data(),DATA_F32,2,{unsigned(legacy.size()),1}},{dummy.data(),DATA_I32,2,{1,b.cap*batch}}},{{reference.data(),DATA_BF16,1,{256,b.cap*batch}}},{1});
  for(unsigned local=0;local<batch;++local)for(unsigned row=0;row<b.cap;++row)for(unsigned n=0;n<256;++n){unsigned index=(local*b.cap+row)*256+n;uint16_t expected=wantmap[slot+local]<0||row>=unsigned(wantcounts[wantmap[slot+local]])?0:reference[index];eq(gate[index],expected);}
 }
 std::cout<<"{\"case\":\""<<kind<<"\",\"checked\":"<<checked<<",\"bad\":"<<bad<<",\"device_validated\":false,\"MME_tested\":false,\"literal_gate_ISA_reference\":true}\n";return bad?3:0;
}
