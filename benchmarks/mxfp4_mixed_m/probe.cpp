#include "../fp8_mme_contract/harness.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static uint16_t bf(float v){uint32_t u;std::memcpy(&u,&v,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t v){uint32_t u=uint32_t(v)<<16;float x;std::memcpy(&x,&u,4);return x;}
static const float q4[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
struct Chunk{int start,count,m,row;};
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("probe padded|pairs|bucket|merge12|exact|fused|bounded|independent|lean N K n1 n2 n3 banks");
 std::string policy=argv[1];int N=std::stoi(argv[2]),K=std::stoi(argv[3]),n1=std::stoi(argv[4]),n2=std::stoi(argv[5]),n3=std::stoi(argv[6]),banks=std::stoi(argv[7]);
 int E=n1+n2+n3,pool=E*banks,mode=policy=="padded"?0:policy=="pairs"?1:policy=="bucket"?2:policy=="merge12"?3:policy=="exact"?4:policy=="fused"?5:policy=="bounded"?6:policy=="independent"?7:policy=="lean"?8:-1;
 if(mode<0||E<8||E>128||n1<0||n2<0||n3<0||banks<1||pool>384||N<256||N%256||K<32||K%32)throw std::runtime_error("unsupported geometry");
 const char*up_env=std::getenv("GK_MIXED_UPSTREAM");std::string upstream=up_env?up_env:"none";bool has_upstream=upstream!="none",fuse_upstream=upstream=="fused";if(has_upstream&&(mode!=8||(upstream!="separate"&&upstream!="fused")))throw std::runtime_error("upstream test requires lean and separate|fused");
 bool synthetic=std::getenv("GK_MIXED_SYNTHETIC_PRODUCER")!=nullptr;
 bool rotate_counts=std::getenv("GK_MIXED_ROTATE_COUNTS")!=nullptr;
 if(rotate_counts&&mode!=0&&mode!=5&&mode!=6&&mode!=7&&mode!=8)throw std::runtime_error("changing counts require the bounded M3 recipe; bucket capacities are shape-specific");
 std::vector<int> counts,order(E),caps(E),inverse(E),rowoff(E);int left[]={n1,n2,n3};
 while(int(counts.size())<E)for(int j=0;j<3;++j)if(left[j]){counts.push_back(j+1);--left[j];}
 std::iota(order.begin(),order.end(),0);
 if(mode==2||mode==3)std::stable_sort(order.begin(),order.end(),[&](int a,int b){return (mode==2?counts[a]:(counts[a]==3?3:2))<(mode==2?counts[b]:(counts[b]==3?3:2));});
 for(int s=0;s<E;++s){inverse[order[s]]=s;caps[s]=(mode==0||mode==5||mode==6||mode==7||mode==8)?3:mode==3?(counts[order[s]]==3?3:2):counts[order[s]];}
 if(mode==1)for(int s=0;s<E;s+=2){int c=caps[s];if(s+1<E)c=std::max(c,caps[s+1]);caps[s]=c;if(s+1<E)caps[s+1]=c;}
 int slots=0;for(int s=0;s<E;++s){rowoff[s]=slots;slots+=caps[s];}
 std::vector<Chunk>chunks;
 for(int s=0;s<E;){int c=1;if(mode!=4&&s+1<E&&caps[s+1]==caps[s])c=2;chunks.push_back({s,c,caps[s],rowoff[s]});s+=c;}
 Harness h;uint64_t wb=uint64_t(pool)*N*K/2,sb=uint64_t(pool)*N*K/32,xb=uint64_t(E)*3*K*2,yb=uint64_t(N)*slots*4;
 auto w=h.tensor("packed",syn_type_uint8,{128,K,N/256,pool},true,wb);auto*wp=(uint8_t*)h.buffers.back().host;
 auto sc=h.tensor("scales",syn_type_uint8,{256,K/32,N/256,pool},true,sb);auto*sp=(uint8_t*)h.buffers.back().host;
 for(int e=0;e<pool;++e)for(int nb=0;nb<N/256;++nb){
  for(int k=0;k<K;++k)for(int j=0;j<128;++j){auto q=[&](int n){return mix(n*31337+k*97+e*131)&15;};wp[((uint64_t(e)*(N/256)+nb)*K+k)*128+j]=q(nb*256+j)|(q(nb*256+j+128)<<4);}
  for(int g=0;g<K/32;++g)for(int j=0;j<256;++j)sp[((uint64_t(e)*(N/256)+nb)*(K/32)+g)*256+j]=118+mix(nb*65537+j*317+g*43+e*103)%15;
 }
 auto lut=h.tensor("byte_lut",syn_type_bf16,{512,1},true,1024);auto*lp=(uint16_t*)h.buffers.back().host;for(int i=0;i<256;++i){lp[2*i]=bf(q4[i&15]);lp[2*i+1]=bf(q4[i>>4]);}
 int activation_banks=rotate_counts?banks:1;uint64_t xstride=(xb+255)/256*256,cstride=(E*4+255)/256*256;
 std::vector<uint8_t>xexpected(has_upstream?xstride*activation_banks:0);auto x=h.tensor("activation",syn_type_bf16,{K,3,E},!has_upstream,xstride*activation_banks);auto*xbytes=has_upstream?xexpected.data():(uint8_t*)h.buffers.back().host;
 auto c=h.tensor("counts",syn_type_int32,{E,1},true,cstride*activation_banks);auto*cbytes=(uint8_t*)h.buffers.back().host;
 for(int b=0;b<activation_banks;++b){auto*xp=(uint16_t*)(xbytes+b*xstride);auto*cp=(int32_t*)(cbytes+b*cstride);
  for(int e=0;e<E;++e){cp[e]=rotate_counts?1+(counts[e]-1+b)%3:counts[e];for(int m=0;m<3;++m)for(int k=0;k<K;++k)xp[(uint64_t(e)*3+m)*K+k]=m<cp[e]?bf((int(mix(m*65537+k*131+e*997+b*2341)%2049)-1024)/1024.f):0x7fc1;}
 }
 synTensor tokens=nullptr,row_ids=nullptr;uint64_t token_stride=uint64_t(K)*16*2;
 if(has_upstream){
  tokens=h.tensor("tokens",syn_type_bf16,{K,16},true,token_stride*activation_banks);auto*tp=(uint16_t*)h.buffers.back().host;
  row_ids=h.tensor("token_rows",syn_type_int32,{3,E},true,E*3*4);auto*rp=(int32_t*)h.buffers.back().host;
  for(int e=0;e<E;++e)for(int m=0;m<3;++m)rp[e*3+m]=(e*3*13+m*7+5)%16;
  for(int b=0;b<activation_banks;++b){auto*xp=(uint16_t*)(xbytes+b*xstride);auto*cp=(int32_t*)(cbytes+b*cstride);
   for(int t=0;t<16;++t)for(int k=0;k<K;++k)tp[(uint64_t(b)*16+t)*K+k]=bf((int(mix(t*65537+k*131+b*2341)%2049)-1024)/1024.f);
   for(int e=0;e<E;++e)for(int m=0;m<3;++m)for(int k=0;k<K;++k)xp[(uint64_t(e)*3+m)*K+k]=m<cp[e]?tp[(uint64_t(b)*16+rp[e*3+m])*K+k]:0x7fc1;
  }
  save("activation.bin",xbytes,xstride*activation_banks);
 }
 auto cap=h.tensor("capacities",syn_type_int32,{E,1},true,E*4);std::memcpy(h.buffers.back().host,caps.data(),E*4);
 uint64_t stride=((E*4+255)/256)*256;auto ids=h.tensor("expert_ids",syn_type_int32,{E,1},true,stride*banks);auto*ip=(uint8_t*)h.buffers.back().host;
 std::vector<int>perm(pool);std::iota(perm.begin(),perm.end(),0);std::sort(perm.begin(),perm.end(),[](int a,int b){return mix(a+3571)<mix(b+3571);});for(int b=0;b<banks;++b)for(int e=0;e<E;++e)((int32_t*)(ip+b*stride))[e]=perm[b*E+e];
 synTensor status=nullptr;int32_t*flags=nullptr;if(mode!=5){status=h.tensor("capacity_status",syn_type_int32,{E,1},true,E*4,true);flags=(int32_t*)h.buffers.back().host;}
 auto y=h.tensor("output",syn_type_float,{N,slots},true,yb,true);auto*yp=(float*)h.buffers.back().host;
 synTensor ap=x,pi=ids;
 if(mode==8){ap=h.tensor("prepared_activation",syn_type_bf16,{K,slots},false);
  if(has_upstream){int flat=fuse_upstream?1:0;auto check=fuse_upstream?status:h.tensor("gather_status",syn_type_int32,{E,1},false);h.node("gk_mixed_m_gather","upstream_gather",{tokens,row_ids,c},{fuse_upstream?ap:x,check},&flat,4);}
  if(!fuse_upstream)h.node("gk_mixed_m_prepare_lean","mixed_prepare",{x,c},{ap,status});}
 else if(mode!=5){ap=h.tensor("prepared_activation",syn_type_bf16,{K,slots},false);pi=h.tensor("prepared_ids",syn_type_int32,{E,1},false);h.node(mode>=6?"gk_mixed_m_prepare_bounded":"gk_mixed_m_prepare","mixed_prepare",{x,c,ids,cap},{ap,pi,status},&mode,4);}
 if(mode==7)pi=ids; // Weight decode does not depend on activation preparation.
 std::vector<synTensor>av,iv,yv,cv;
 for(unsigned i=0;i<chunks.size();++i){auto q=chunks[i];std::string tag="mixed_"+std::to_string(i);av.push_back(h.tensor(tag+"_a_flat",syn_type_bf16,mode==5?std::vector<int>{K,3,q.count}:std::vector<int>{K,q.m*q.count},false));iv.push_back(h.tensor(tag+"_ids",syn_type_int32,{q.count,1},false));yv.push_back(h.tensor(tag+"_y_flat",syn_type_float,{N,q.m*q.count},false));if(mode==5)cv.push_back(h.tensor(tag+"_counts",syn_type_int32,{q.count,1},false));}
 synSplitParams sa{mode==5?2u:1u},si{0};h.node("split","activation_views",{ap},av,&sa,sizeof(sa));h.node("split","id_views",{pi},iv,&si,sizeof(si));if(mode==5)h.node("split","count_views",{c},cv,&si,sizeof(si));
 for(unsigned i=0;i<chunks.size();++i){auto q=chunks[i];std::string tag="mixed_"+std::to_string(i);auto a=h.tensor(tag+"_a",syn_type_bf16,{K,q.m,q.count},false),out=h.tensor(tag+"_y",syn_type_float,{N,q.m,q.count},false),d=h.tensor("decoded_weights_"+std::to_string(i),syn_type_bf16,{N,K,q.count},false);
  if(mode==5)h.node("gk_mixed_m_decode_prepare",(tag+"_decode").c_str(),{w,sc,lut,iv[i],av[i],cv[i]},{d,a});
  else{h.node("reshape",(tag+"_a_view").c_str(),{av[i]},{a});h.node(synthetic?"gk_mixed_m_fill":"gk_grouped_mxfp4_decode",(tag+"_decode").c_str(),{w,sc,lut,iv[i]},{d});}
  synGEMMParams gp{false,false};h.node("batch_gemm",(tag+"_mme").c_str(),{a,d},{out},&gp,sizeof(gp));h.node("reshape",(tag+"_y_view").c_str(),{out},{yv[i]});
 }
 synConcatenateParams cat{1};h.node("concat","output_views",yv,{y},&cat,sizeof(cat));
 std::printf("{\"stage\":\"plan\",\"policy\":\"%s\",\"experts\":%d,\"N\":%d,\"K\":%d,\"n1\":%d,\"n2\":%d,\"n3\":%d,\"banks\":%d,\"pool\":%d,\"useful_rows\":%d,\"row_slots\":%d,\"planned_mme_nodes\":%zu,\"original_bytes\":%llu,\"rotating_bytes\":%llu,\"counts_device_resident\":true,\"invalid_input_rows\":\"BF16 NaN, must be masked\"}\n",policy.c_str(),E,N,K,n1,n2,n3,banks,pool,n1+2*n2+3*n3,slots,chunks.size(),(unsigned long long)((wb+sb)/banks),(unsigned long long)(wb+sb));std::fflush(stdout);
 std::printf("{\"stage\":\"diagnostic\",\"synthetic_producer\":%s}\n",synthetic?"true":"false");
 std::printf("{\"stage\":\"upstream\",\"mode\":\"%s\"}\n",upstream.c_str());
 save("slot_to_expert.bin",order.data(),E*4);save("row_offsets.bin",rowoff.data(),E*4);
 std::printf("{\"stage\":\"runtime_inputs\",\"changing_counts\":%s,\"activation_banks\":%d,\"activation_stride\":%llu,\"counts_stride\":%llu,\"id_stride\":%llu}\n",rotate_counts?"true":"false",activation_banks,(unsigned long long)xstride,(unsigned long long)cstride,(unsigned long long)stride);std::fflush(stdout);
 check(synGraphCompile(&h.recipe,h.graph,"mixed_m",nullptr),"compile");if(std::getenv("GK_MXFP4_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\"}");h.close();return 0;}
 h.prepare();size_t id_index=0,x_index=0,c_index=0,token_index=0;for(unsigned i=0;i<h.launches.size();++i){std::string name=h.launches[i].tensorName;if(name=="expert_ids")id_index=i;else if(name=="activation")x_index=i;else if(name=="counts")c_index=i;else if(name=="tokens")token_index=i;}uint64_t id_base=h.launches[id_index].pTensorAddress,x_base=h.launches[x_index].pTensorAddress,c_base=h.launches[c_index].pTensorAddress,token_base=h.launches[token_index].pTensorAddress;
 auto run=[&](int bank){h.launches[id_index].pTensorAddress=id_base+bank*stride;int ab=rotate_counts?bank:0;if(has_upstream)h.launches[token_index].pTensorAddress=token_base+ab*token_stride;else h.launches[x_index].pTensorAddress=x_base+ab*xstride;h.launches[c_index].pTensorAddress=c_base+ab*cstride;h.run();};
 size_t bad=0,checked=0;double err2=0,ref2=0,maxback=0;std::vector<double>reference(uint64_t(N)*slots);
 for(int bank=0;bank<banks;++bank){h.poison();run(bank);h.download("_bank"+std::to_string(bank));for(int e=0;e<E;++e)if(flags&&flags[e])throw std::runtime_error("device capacity gate failed");auto*selected=(int32_t*)(ip+bank*stride);int ab=rotate_counts?bank:0;auto*xp=(uint16_t*)(xbytes+ab*xstride);auto*cp=(int32_t*)(cbytes+ab*cstride);
  std::fill(reference.begin(),reference.end(),0);
  for(int e=0;e<E;++e)for(int n=0;n<N;++n){int source=selected[e],slot=inverse[e];std::vector<double>wk(K);for(int k=0;k<K;++k){unsigned byte=wp[((uint64_t(source)*(N/256)+n/256)*K+k)*128+n%128],q=(byte>>((n%256>=128)*4))&15,s=sp[((uint64_t(source)*(N/256)+n/256)*(K/32)+k/32)*256+n%256];wk[k]=synthetic?1+(source&3):std::ldexp(double(q4[q]),int(s)-127);}
   for(int m=0;m<caps[slot];++m){double ref=0,absolute=0;if(m<cp[e])for(int k=0;k<K;++k){double p=wk[k]*fp(xp[(uint64_t(e)*3+m)*K+k]);ref+=p;absolute+=std::abs(p);}size_t at=uint64_t(rowoff[slot]+m)*N+n;double delta=std::abs(double(yp[at])-ref);reference[at]=ref;bad+=!std::isfinite(yp[at])||delta>2e-5+2e-6*absolute;++checked;err2+=delta*delta;ref2+=ref*ref;maxback=std::max(maxback,delta/std::max(absolute,1e-30));}
  }save("oracle_bank"+std::to_string(bank)+".bin",reference.data(),reference.size()*8);
 }
 std::printf("{\"stage\":\"correctness\",\"bad\":%zu,\"checked\":%zu,\"relative_l2\":%.12g,\"max_componentwise_backward\":%.12g,\"workspace_bytes\":%llu}\n",bad,checked,std::sqrt(err2/std::max(ref2,1e-30)),maxback,(unsigned long long)h.workspace_bytes);std::fflush(stdout);if(bad)throw std::runtime_error("full output oracle failed");
 for(int i=0;i<8;++i)run(i%banks);check(synStreamSynchronize(h.stream),"warmup");synEventHandle begin,end;check(synEventCreate(&begin,h.dev,EVENT_COLLECT_TIME),"event");check(synEventCreate(&end,h.dev,EVENT_COLLECT_TIME),"event");
 int reps=std::getenv("GK_TIMING_REPEATS")?std::stoi(std::getenv("GK_TIMING_REPEATS")):20;if(reps<1||reps>500)throw std::runtime_error("timing repeats 1..500");std::printf("{\"stage\":\"timing_config\",\"repeats\":%d}\n",reps);for(int sample=0;sample<5;++sample){auto start=std::chrono::steady_clock::now();check(synEventRecord(begin,h.stream),"begin");for(int i=0;i<reps;++i)run(i%banks);check(synEventRecord(end,h.stream),"end");check(synEventSynchronize(end),"wait");uint64_t ns;check(synEventElapsedTime(&ns,begin,end),"elapsed");double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/reps;std::printf("{\"stage\":\"timing\",\"sample\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,double(ns)/1000/reps,wall);}
 h.close();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"mixed_m: %s\n",e.what());return 1;}}
