#include "harness.hpp"
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("codes a|b|a4|b4|a44|b44");std::string mode=argv[1];bool a=mode=="a"||mode=="a4"||mode=="a44",e4=mode.size()>1,both=mode.size()==3;if(!a&&mode!="b"&&mode!="b4"&&mode!="b44")return 2;
 const int E=8,K=256,N=a?128:256,M=a?256:1;
 std::vector<uint8_t>codes={0,128};for(int i=e4?1:4;i<(e4?120:124);++i){codes.push_back(i);codes.push_back(i|128);}while(codes.size()<256)codes.push_back(0);
 Harness h;auto x=h.tensor("activation",e4&&(a||both)?syn_type_fp8_143:syn_type_fp8_152,{K,M,E},true,uint64_t(E)*K*M);auto*xp=(uint8_t*)h.buffers.back().host;
 auto w=h.tensor("weight",e4&&(!a||both)?syn_type_fp8_143:syn_type_fp8_152,{N,K,E},true,uint64_t(E)*N*K);auto*wp=(uint8_t*)h.buffers.back().host;
 auto y=h.tensor("output",syn_type_single,{N,M,E},true,uint64_t(E)*N*M*4,true);auto*yp=(float*)h.buffers.back().host;
 for(int e=0;e<E;++e){for(int m=0;m<M;++m)xp[(uint64_t(e)*M+m)*K+(e*31)%K]=a?codes[m]:(both?56:60);for(int n=0;n<N;++n)wp[(uint64_t(e)*K+(e*31)%K)*N+n]=a?(both?56:60):codes[n];}
 synGEMMParams p{false,false};h.node("batch_gemm","native_fp8_codes",{x,w},{y},&p,sizeof(p));
 std::printf("{\"stage\":\"plan\",\"mode\":\"codes-%s\",\"experts\":8,\"normal_codes_and_zeros\":%d,\"N\":%d,\"K\":%d,\"M\":%d}\n",argv[1],e4?226:242,N,K,M);std::fflush(stdout);h.compile();if(h.compile_only()){h.close();return 0;}h.prepare();h.poison();h.run();h.download("_checked");
 size_t bad=0,checked=0;std::vector<double>ref(uint64_t(E)*N*M);
 for(int e=0;e<E;++e)for(int m=0;m<M;++m)for(int n=0;n<N;++n){size_t at=(uint64_t(e)*M+m)*N+n;uint8_t c=codes[a?m:n];int ex=(c>>3)&15;ref[at]=e4?std::copysign(ex?std::ldexp(double(8+(c&7)),ex-10):std::ldexp(double(c&7),-9),c&128?-1.:1.):h8(c);bad+=!std::isfinite(yp[at])||double(yp[at])!=ref[at];++checked;}
 save("reference_f64.bin",ref.data(),ref.size()*8);std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"workspace_bytes\":%llu}\n",checked,bad,(unsigned long long)h.workspace_bytes);h.close();return bad?3:0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
