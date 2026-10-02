#include "../fp8_mme_contract/harness.hpp"
#include <chrono>
int main(int argc,char** argv) { try {
    if(argc!=3&&argc!=4) throw std::runtime_error("usage: feed bf16|fp8 8|32 [shared-a]");
    std::string mode=argv[1]; int E=std::stoi(argv[2]);
    bool shared=argc==4; if(shared&&std::string(argv[3])!="shared-a") throw std::runtime_error("unknown layout");
    if((mode!="bf16"&&mode!="fp8")||(E!=8&&E!=32)) throw std::runtime_error("invalid shape or mode");
    constexpr int N=512,K=6144,M=2; bool fp8=mode=="fp8"; int bytes=fp8?1:2;
    auto dtype=fp8?syn_type_fp8_143:syn_type_bf16; Harness h;
    int graphN=shared?N*E:N, graphE=shared?1:E;
    auto a=h.tensor("activation",dtype,{K,M,graphE},true,uint64_t(graphE)*M*K*bytes);
    auto seed=h.tensor("seed",dtype,{graphN,graphE},true,uint64_t(E)*N*bytes);
    auto w=h.tensor("generated_weight",dtype,{graphN,K,graphE},false);
    auto y=h.tensor("output",syn_type_float,{graphN,M,graphE},true,uint64_t(E)*M*N*4,true);
    const float values[]={0,.5f,1,1.5f,2,3,4,6}; const uint8_t codes[]={0,48,56,60,64,68,72,76};
    auto bf16=[](float v){uint32_t u;std::memcpy(&u,&v,4);return uint16_t(u>>16);};
    for(int e=0;e<graphE;++e) for(int m=0;m<M;++m) for(int k=0;k<K;++k) {
        size_t i=(e*M+m)*K+k;
        if(fp8) ((uint8_t*)h.buffers[0].host)[i]=(m?64:56)^((k&1)?128:0);
        else ((uint16_t*)h.buffers[0].host)[i]=bf16(float(m+1)*((k&1)?-1.f:1.f));
    }
    std::vector<float> expected(E*M*N);
    for(int e=0;e<E;++e) for(int n=0;n<N;++n) {
        int code=(n+e*3)%8; bool sign=((n/8+e)&1)!=0; float v=values[code]*(sign?-1.f:1.f);
        if(fp8) ((uint8_t*)h.buffers[1].host)[e*N+n]=codes[code]^(sign?128:0);
        else ((uint16_t*)h.buffers[1].host)[e*N+n]=bf16(v);
        for(int m=0;m<M;++m) expected[(shared?m*E+e:e*M+m)*N+n]=v*K*(m+1);
    }
    h.node(fp8?"gk_generated_feed_fp8":"gk_generated_feed_bf16","generate_weight",{seed},{w});
    synGEMMParams params{false,false}; h.node("batch_gemm","consume_weight",{a,w},{y},&params,sizeof(params));
    std::printf("{\"stage\":\"plan\",\"mode\":\"%s\",\"E\":%d,\"N\":%d,\"K\":%d,\"M\":%d,\"layout\":\"%s\",\"graph_N\":%d,\"graph_E\":%d,\"weight_element_bytes\":%d,\"generated_weight_bytes\":%llu,\"mxfp4_equivalent_bytes\":%llu,\"real_mxfp4\":false,\"activation_quantization_included\":false}\n",mode.c_str(),E,N,K,M,shared?"shared-a-n":"expert-batch",graphN,graphE,bytes,(unsigned long long)E*N*K*bytes,(unsigned long long)E*N*K*17/32);
    std::fflush(stdout); h.compile();
    if(std::getenv("GK_MXFP4_COMPILE_ONLY")) {std::puts("{\"stage\":\"compile_only\",\"device_launch\":false}");h.close();return 0;}
    h.prepare();h.poison();h.run();h.download("_checked");
    size_t bad=0; float max_abs=0; const float* actual=(float*)h.buffers[2].host;
    for(size_t i=0;i<expected.size();++i) { if(!std::isfinite(actual[i])||actual[i]!=expected[i]) ++bad; max_abs=std::fmax(max_abs,std::fabs(actual[i]-expected[i])); }
    std::printf("{\"stage\":\"correctness\",\"checked\":%zu,\"bad\":%zu,\"max_abs\":%.9g,\"oracle\":\"exact seed times K times row\"}\n",expected.size(),bad,max_abs);
    if(bad) throw std::runtime_error("full output gate failed");
    for(int i=0;i<8;++i) h.run(); check(synStreamSynchronize(h.stream),"warmup sync");
    synEventHandle begin,end;check(synEventCreate(&begin,h.dev,EVENT_COLLECT_TIME),"event");check(synEventCreate(&end,h.dev,EVENT_COLLECT_TIME),"event");
    constexpr int repeats=20;
    for(int sample=0;sample<5;++sample) {
        auto start=std::chrono::steady_clock::now();check(synEventRecord(begin,h.stream),"event begin");
        for(int i=0;i<repeats;++i) h.run();check(synEventRecord(end,h.stream),"event end");check(synEventSynchronize(end),"event sync");
        uint64_t ns;check(synEventElapsedTime(&ns,begin,end),"elapsed");
        double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        std::printf("{\"stage\":\"timing\",\"sample\":%d,\"repeats\":%d,\"event_us\":%.6f,\"wall_us\":%.6f}\n",sample,repeats,double(ns)/1000/repeats,wall);
    }
    h.close();return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"feed control: %s\n",e.what());return 1; } }
