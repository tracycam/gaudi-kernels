// SPDX-License-Identifier: Apache-2.0
// Compare both split terms with the existing native decoder and scalar FP32 math.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <synapse_common_types.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace gaudirt::runtime;
namespace {
void check(synStatus s) { if(s!=synSuccess) throw std::runtime_error("Synapse status "+std::to_string(s)); }
float widen(uint16_t x) { uint32_t bits=uint32_t(x)<<16; float f; std::memcpy(&f,&bits,4); return f; }
float fp8(uint8_t x) {
  unsigned e=(x>>3)&15, m=x&7; if(e==15 && m==7) return NAN;
  float v=e==0 ? std::ldexp(float(m),-9) : std::ldexp(1.0f+float(m)/8.0f,int(e)-7);
  return x&128 ? -v : v;
}
template<class T> std::vector<T> read(const std::filesystem::path& p) {
  auto bytes=std::filesystem::file_size(p); if(bytes%sizeof(T)) throw std::runtime_error("bad file size");
  std::vector<T> data(bytes/sizeof(T)); std::ifstream f(p,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(data.data()),bytes)) throw std::runtime_error("read failed");
  return data;
}
}
int main(int argc,char** argv) {
  if(argc!=3) return 2;
  try {
    const auto rank=std::stoi(argv[2]); DeviceRuntime::Config cfg{}; cfg.module_id=rank;
    DeviceRuntime rt(cfg); HbmArena arena(rt.device_id(),512ULL*1024*1024); WeightManifest weights(argv[1],rt,arena);
    constexpr uint64_t N=6144,G=16,K=G*128;
    synGraphHandle graph=nullptr; synRecipeHandle recipe=nullptr; check(synGraphCreate(&graph,synDeviceGaudi2));
    std::array<synTensor,5> t{}; std::array<synSectionHandle,5> sections{}; std::array<synLaunchTensorInfo,5> launch{};
    auto make=[&](int i,const char* name,synDataType type,std::initializer_list<uint64_t> shape,uint64_t addr) {
      check(synSectionCreate(&sections[i],0,graph)); check(synSectionSetPersistent(sections[i],true));
      synTensorDescriptor d{};d.m_name=name;d.m_dataType=type;d.m_dims=shape.size();unsigned j=0;
      launch[i].tensorName=name;launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=addr;
      for(auto n:shape) { d.m_sizes[j]=d.m_minSizes[j]=n; launch[i].tensorSize[j++]=n; }
      check(synTensorCreate(&t[i],&d,sections[i],0));
      if(type==syn_type_fp8_143) { synFpQuantParam p{1.0,7};synFpQuantMetadata m{type,&p,1};check(synTensorSetQuantizationData(t[i],SYN_FP_QUANT_METADATA,&m,sizeof(m))); }
    };
    const auto& w=weights.get("down_weight"); const auto& scales=weights.get("down_scales");
    make(0,"weight",syn_type_fp8_143,{128,N,G},w.device_address); make(1,"scales",syn_type_float,{G,N/128},scales.device_address);
    const uint64_t bytes=N*K*2;
    make(2,"high",syn_type_bf16,{K,N},arena.allocate(bytes)); make(3,"low",syn_type_bf16,{K,N},arena.allocate(bytes)); make(4,"existing",syn_type_bf16,{K,N},arena.allocate(bytes));
    synTensor input[]={t[0],t[1]}, split[]={t[2],t[3]};
    check(synNodeCreate(graph,input,split,2,2,nullptr,0,"gk_dense_block128_decode_split_v1","split",nullptr,nullptr));
    check(synNodeCreate(graph,input,&t[4],2,1,nullptr,0,"gk_block128_decode_v1","existing",nullptr,nullptr));
    check(synGraphCompile(&recipe,graph,"dense_weight_split_test",nullptr));
    uint64_t ws=0;check(synWorkspaceGetSize(&ws,recipe));initialize_launch_tensor_ids(recipe,launch.data(),launch.size());
    check(synGraphDestroy(graph));for(auto sec:sections) check(synSectionDestroy(sec));
    PreparedLaunch prepared{recipe,arena.ensure_workspace(ws),{launch.begin(),launch.end()},0};prepared.execute(rt.compute_stream());rt.synchronize_compute();
    std::vector<uint16_t> hi(N*K),lo(N*K),ref(N*K);
    arena.copy_d2h_sync(rt.d2h_stream(),launch[2].pTensorAddress,hi.data(),bytes);
    arena.copy_d2h_sync(rt.d2h_stream(),launch[3].pTensorAddress,lo.data(),bytes);
    arena.copy_d2h_sync(rt.d2h_stream(),launch[4].pTensorAddress,ref.data(),bytes);
    const auto packed=read<uint8_t>(std::filesystem::path(argv[1])/"down_weight.bin");
    const auto sc=read<float>(std::filesystem::path(argv[1])/"down_scales.bin");
    if(packed.size()!=N*K || sc.size()!=N*G/128) throw std::runtime_error("bad fixture");
    uint64_t mismatches=0;double err=0,energy=0,max_abs=0;uint64_t nfinite=0;
    for(uint64_t n=0;n<N;++n) for(uint64_t g=0;g<G;++g) for(uint64_t k=0;k<128;++k) {
      const auto out=n*K+g*128+k, input=(g*N+n)*128+k;
      mismatches+=hi[out]!=ref[out];float expected=fp8(packed[input])*sc[(n/128)*G+g];
      float actual=widen(hi[out])+widen(lo[out]); if(!std::isfinite(expected)||!std::isfinite(actual)) {++nfinite;continue;}
      double diff=double(actual)-expected;err+=diff*diff;energy+=double(expected)*expected;max_abs=std::max(max_abs,std::abs(diff));
    }
    const double rel=std::sqrt(err/std::max(energy,1e-300));
    std::cout<<"values="<<N*K<<" high_bitwise_mismatches="<<mismatches<<" nonfinite="<<nfinite<<" relative_l2="<<rel<<" max_abs="<<max_abs<<"\n";
    check(synRecipeDestroy(recipe));if(mismatches||nfinite||rel>1e-5) return 1;return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
