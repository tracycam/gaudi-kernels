#include "../fp8_mme_contract/harness.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static uint16_t bf(float v){uint32_t u;std::memcpy(&u,&v,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static const float q4[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
struct Matrix{synTensor w,s;uint8_t*wp,*sp;int N,K;};
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("probe stage|prefetch1|prefetch2|stream banks");
 std::string mode=argv[1];int banks=std::stoi(argv[2]);bool stream=mode=="stream";int early=mode=="prefetch1"?1:mode=="prefetch2"?2:0;
 if((mode!="stage"&&!stream&&!early)||banks<1||banks>4)throw std::runtime_error("unsupported chain");
 const char*ug=std::getenv("GK_CHAIN_GATHER");std::string upstream=ug?ug:"none";bool gathered=upstream!="none",fused=upstream=="fused";if(gathered&&upstream!="separate"&&!fused)throw std::runtime_error("GK_CHAIN_GATHER requires separate|fused");
 const int E=20,C=10,K=6144,H=256,T=5;int pool=E*banks;
 Harness h;h.buffers.reserve(64);
 auto matrix=[&](std::string name,int n,int k,int salt){Matrix a;a.N=n;a.K=k;a.w=h.tensor(name+"_packed",syn_type_uint8,{128,k,n/256,pool},true,uint64_t(pool)*n*k/2);a.wp=(uint8_t*)h.buffers.back().host;a.s=h.tensor(name+"_scales",syn_type_uint8,{256,k/32,n/256,pool},true,uint64_t(pool)*n*k/32);a.sp=(uint8_t*)h.buffers.back().host;
  for(int e=0;e<pool;++e)for(int nb=0;nb<n/256;++nb){for(int kk=0;kk<k;++kk)for(int j=0;j<128;++j){auto q=[&](int nn){return mix(nn*31337+kk*97+e*131+salt)&15;};a.wp[((uint64_t(e)*(n/256)+nb)*k+kk)*128+j]=q(nb*256+j)|(q(nb*256+j+128)<<4);}
   for(int g=0;g<k/32;++g)for(int j=0;j<256;++j)a.sp[((uint64_t(e)*(n/256)+nb)*(k/32)+g)*256+j]=119+mix(nb*65537+j*317+g*43+e*103+salt)%5;}
  return a;};
 auto gp=matrix("gp",2*H,K,71),down=matrix("down",K,H,3571);
 auto lut=h.tensor("byte_lut",syn_type_bf16,{512,1},true,1024);auto*lp=(uint16_t*)h.buffers.back().host;for(int i=0;i<256;++i){lp[2*i]=bf(q4[i&15]);lp[2*i+1]=bf(q4[i>>4]);}
 std::vector<int> counts(E),rowmap(3*E,-1);for(int e=0;e<E;++e)counts[e]=1+e%3;counts.back()=3;
 int cursor=0;std::vector<int>routes(T);for(int e=0;e<E;++e)for(int m=0;m<counts[e];++m){rowmap[e*3+m]=cursor++%T;++routes[rowmap[e*3+m]];}if(cursor!=40||!std::all_of(routes.begin(),routes.end(),[](int n){return n==8;}))throw std::runtime_error("routing fixture");
 auto ct=h.tensor("counts",syn_type_int32,{E,1},true,E*4);std::memcpy(h.buffers.back().host,counts.data(),E*4);
 auto rt=h.tensor("token_rows",syn_type_int32,{3,E},true,E*3*4);std::memcpy(h.buffers.back().host,rowmap.data(),E*3*4);
 std::vector<uint16_t>xexpected(gathered?K*3*E:0);auto x=h.tensor("activation",syn_type_bf16,{K,3,E},!gathered,uint64_t(K)*3*E*2);auto*xp=gathered?xexpected.data():(uint16_t*)h.buffers.back().host;
 for(int e=0;e<E;++e)for(int m=0;m<3;++m)for(int k=0;k<K;++k)xp[(e*3+m)*K+k]=m<counts[e]?bf((int(mix(rowmap[e*3+m]*65537+k*131)%2049)-1024)/8192.f):0x7fc1;
 synTensor tokens=nullptr;if(gathered){tokens=h.tensor("tokens",syn_type_bf16,{K,T},true,uint64_t(K)*T*2);auto*tp=(uint16_t*)h.buffers.back().host;for(int t=0;t<T;++t)for(int k=0;k<K;++k)tp[t*K+k]=bf((int(mix(t*65537+k*131)%2049)-1024)/8192.f);save("activation.bin",xp,uint64_t(K)*3*E*2);}
 uint64_t stride=256;auto ids=h.tensor("expert_ids",syn_type_int32,{E,1},true,stride*banks);auto*ip=(uint8_t*)h.buffers.back().host;std::vector<int>perm(pool);std::iota(perm.begin(),perm.end(),0);std::sort(perm.begin(),perm.end(),[](int a,int b){return mix(a+977)<mix(b+977);});for(int b=0;b<banks;++b)for(int e=0;e<E;++e)((int32_t*)(ip+b*stride))[e]=perm[b*E+e];
 auto status=h.tensor("capacity_status",syn_type_int32,{E,1},true,E*4,true);auto*flags=(int32_t*)h.buffers.back().host;
 auto gpout=h.tensor("gp_result",syn_type_float,{2*H,3,E},true,uint64_t(2*H)*3*E*4,true);auto*gpp=(float*)h.buffers.back().host;
 auto ga=h.tensor("gate_result",syn_type_bf16,{H,3,E},true,uint64_t(H)*3*E*2,true);auto*gap=(uint16_t*)h.buffers.back().host;
 auto dout=h.tensor("down_result",syn_type_float,{K,3,E},true,uint64_t(K)*3*E*4,true);auto*dp=(float*)h.buffers.back().host;
 auto out=h.tensor("combined",syn_type_float,{K,T},true,uint64_t(K)*T*4,true);auto*yp=(float*)h.buffers.back().host;
 auto apflat=h.tensor("prepared_activation_flat",syn_type_bf16,{K,3*E},false),ap=h.tensor("prepared_activation",syn_type_bf16,{K,3,E},false);
 if(gathered){int flat=fused?1:0;auto check=fused?status:h.tensor("gather_status",syn_type_int32,{E,1},false);h.node("gk_mixed_m_gather","upstream_gather",{tokens,rt,ct},{fused?apflat:x,check},&flat,4);}
 if(!fused)h.node("gk_mixed_m_prepare_lean","prepare",{x,ct},{apflat,status});h.node("reshape","prepare_view",{apflat},{ap});
 auto split=[&](synTensor t,std::string name,synDataType dt,std::vector<int>dims,int axis){std::vector<synTensor>v;for(int i=0;i<C;++i)v.push_back(h.tensor(name+"_"+std::to_string(i),dt,dims,false));synSplitParams p{unsigned(axis)};h.node("split",(name+"_views").c_str(),{t},v,&p,sizeof(p));return v;};
 auto iv=split(ids,"ids",syn_type_int32,{2,1},0),cv=split(ct,"counts",syn_type_int32,{2,1},0),av=split(ap,"a",syn_type_bf16,{K,3,2},2);
 std::vector<synTensor>gy,gatev,dy,readyv,dweights(C,nullptr);auto ready=h.tensor("ready_ids",syn_type_int32,{E,1},false);
 if(!stream){gatev=split(ga,"down_a",syn_type_bf16,{H,3,2},2);readyv=split(ready,"down_ids",syn_type_int32,{2,1},0);}
 auto decode=[&](Matrix&w,synTensor ids,std::string name){auto d=h.tensor(name+"_weights",syn_type_bf16,{w.N,w.K,2},false);h.node("gk_grouped_mxfp4_decode",(name+"_decode").c_str(),{w.w,w.s,lut,ids},{d});return d;};
 auto gemm=[&](synTensor a,synTensor w,std::string name,int n){auto y=h.tensor(name+"_y",syn_type_float,{n,3,2},false);synGEMMParams p{false,false};h.node("batch_gemm",(name+"_mme").c_str(),{a,w},{y},&p,sizeof(p));return y;};
 for(int i=0;i<C;++i){std::string tag=std::to_string(i);gy.push_back(gemm(av[i],decode(gp,iv[i],"gp_"+tag),"gp_"+tag,2*H));
  if(!stream&&i==1)for(int j=0;j<early;++j)dweights[j]=decode(down,iv[j],"down_"+std::to_string(j));
  if(stream){dweights[i]=decode(down,iv[i],"down_"+tag);auto a=h.tensor("gate_"+tag,syn_type_bf16,{H,3,2},false),r=h.tensor("ready_"+tag,syn_type_int32,{2,1},false);h.node("gk_moe_pipeline_gate",("gate_"+tag).c_str(),{gy[i],iv[i],cv[i]},{a,r});gatev.push_back(a);dy.push_back(gemm(a,dweights[i],"down_"+tag,K));}
 }
 synConcatenateParams cat{2};h.node("concat","gp_output_views",gy,{gpout},&cat,sizeof(cat));
 if(stream)h.node("concat","gate_output_views",gatev,{ga},&cat,sizeof(cat));
 else{h.node("gk_moe_pipeline_gate","gate_all",{gpout,ids,ct},{ga,ready});for(int i=0;i<C;++i){if(!dweights[i])dweights[i]=decode(down,readyv[i],"down_"+std::to_string(i));dy.push_back(gemm(gatev[i],dweights[i],"down_"+std::to_string(i),K));}}
 h.node("concat","down_output_views",dy,{dout},&cat,sizeof(cat));h.node("gk_moe_pipeline_combine","combine",{dout,rt,ct},{out});
 std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"experts\":20,\"tokens\":5,\"routes_per_token\":8,\"useful_rows\":40,\"row_slots\":60,\"K\":6144,\"H\":256,\"banks\":%d,\"early_down_groups\":%d,\"original_bytes\":50135040,\"scope\":\"GP-SiLU-down-weighted-combine; upstream stage records token gather mode; persistent diagnostics included in all arms\"}\n",mode.c_str(),banks,early);std::fflush(stdout);
 std::printf("{\"stage\":\"upstream\",\"mode\":\"%s\"}\n",upstream.c_str());std::fflush(stdout);
 check(synGraphCompile(&h.recipe,h.graph,"moe_pipeline",nullptr),"compile");if(std::getenv("GK_MXFP4_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\"}");h.close();return 0;}
 h.prepare();size_t ii=0;for(unsigned i=0;i<h.launches.size();++i)if(std::string(h.launches[i].tensorName)=="expert_ids")ii=i;uint64_t idbase=h.launches[ii].pTensorAddress;auto run=[&](int b){h.launches[ii].pTensorAddress=idbase+b*stride;h.run();};
 auto weight=[&](Matrix&a,int e,int n,int k){unsigned byte=a.wp[((uint64_t(e)*(a.N/256)+n/256)*a.K+k)*128+n%128],q=(byte>>((n%256>=128)*4))&15,s=a.sp[((uint64_t(e)*(a.N/256)+n/256)*(a.K/32)+k/32)*256+n%256];return std::ldexp(double(q4[q]),int(s)-127);};
 for(int bank=0;bank<banks;++bank){h.poison();run(bank);h.download("_bank"+std::to_string(bank));auto*sel=(int32_t*)(ip+bank*stride);size_t badgp=0,badgate=0,baddown=0,badcombine=0;double ge2=0,gr2=0,de2=0,dr2=0;int maxulp=0;
  for(int e=0;e<E;++e){if(flags[e])throw std::runtime_error("capacity gate");for(int m=0;m<3;++m){for(int n=0;n<2*H;++n){double ref=0,abs=0;if(m<counts[e])for(int k=0;k<K;++k){double p=weight(gp,sel[e],n,k)*fp(xp[(e*3+m)*K+k]);ref+=p;abs+=std::abs(p);}double actual=gpp[(e*3+m)*2*H+n];badgp+=!std::isfinite(actual)||std::abs(actual-ref)>2e-5+2e-6*abs;}
   for(int n=0;n<H;++n){uint16_t ref=0;if(m<counts[e]){float g=fp(bf(gpp[(e*3+m)*2*H+n])),u=fp(bf(gpp[(e*3+m)*2*H+H+n]));float silu=g/(1.f+std::exp(-g));ref=bf(fp(bf(silu))*u);}auto actual=gap[(e*3+m)*H+n];int ulp=std::abs(int(actual)-int(ref));maxulp=std::max(maxulp,ulp);double delta=fp(actual)-fp(ref);ge2+=delta*delta;gr2+=double(fp(ref))*fp(ref);badgate+=!std::isfinite(fp(actual))||ulp>2;}
   for(int n=0;n<K;++n){double ref=0,abs=0;for(int k=0;k<H;++k){double p=weight(down,sel[e],n,k)*fp(gap[(e*3+m)*H+k]);ref+=p;abs+=std::abs(p);}double actual=dp[(e*3+m)*K+n],delta=actual-ref;de2+=delta*delta;dr2+=ref*ref;baddown+=!std::isfinite(actual)||std::abs(delta)>2e-5+2e-6*abs;}
  }}
  std::vector<float>cref(K*T);for(int t=0;t<T;++t)for(int n=0;n<K;++n){float sum=0;for(int e=0;e<E;++e)for(int m=0;m<counts[e];++m)if(rowmap[e*3+m]==t)sum+=dp[(e*3+m)*K+n]*0.125f;cref[t*K+n]=sum;badcombine+=!std::isfinite(yp[t*K+n])||yp[t*K+n]!=sum;}save("combine_oracle_bank"+std::to_string(bank)+".bin",cref.data(),cref.size()*4);
  std::printf("{\"stage\":\"correctness\",\"bank\":%d,\"gp_bad\":%zu,\"gate_bad\":%zu,\"down_bad\":%zu,\"combine_bad\":%zu,\"gate_max_bf16_ulp\":%d,\"gate_relative_l2\":%.12g,\"down_relative_l2\":%.12g}\n",bank,badgp,badgate,baddown,badcombine,maxulp,std::sqrt(ge2/std::max(gr2,1e-30)),std::sqrt(de2/std::max(dr2,1e-30)));std::fflush(stdout);if(badgp||badgate||baddown||badcombine||std::sqrt(ge2/std::max(gr2,1e-30))>0.001)throw std::runtime_error("operator numerical gate");
 }
 for(int i=0;i<8;++i)run(i%banks);check(synStreamSynchronize(h.stream),"warmup");synEventHandle begin,end;check(synEventCreate(&begin,h.dev,EVENT_COLLECT_TIME),"event");check(synEventCreate(&end,h.dev,EVENT_COLLECT_TIME),"event");
 for(int s=0;s<5;++s){auto start=std::chrono::steady_clock::now();check(synEventRecord(begin,h.stream),"begin");for(int i=0;i<20;++i)run(i%banks);check(synEventRecord(end,h.stream),"end");check(synEventSynchronize(end),"wait");uint64_t ns;check(synEventElapsedTime(&ns,begin,end),"elapsed");std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",s,double(ns)/20000,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20);}
 h.close();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"moe_pipeline: %s\n",e.what());return 1;}}
