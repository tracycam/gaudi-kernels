// SPDX-License-Identifier: Apache-2.0
// Check both BF16 split terms against independent round-to-nearest-even math.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <synapse_common_types.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace gaudirt::runtime;
namespace {
void check(synStatus s) { if (s != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(s)); }
uint16_t round_bf16(float x) {
  uint32_t bits; std::memcpy(&bits, &x, 4);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float widen(uint16_t x) { uint32_t bits = uint32_t(x) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows) {
  const size_t n = static_cast<size_t>(width) * rows;
  std::vector<float> input(n);
  for (size_t i=0; i<n; ++i) {
    int32_t a=static_cast<int32_t>((i*48271ULL+17)%200001)-100000;
    input[i]=std::ldexp(float(a), static_cast<int>((i/127)%16)-20) + float(i%257)*0.00000037f;
  }
  const auto x=arena.allocate(n*4), hi=arena.allocate(n*2), lo=arena.allocate(n*2);
  arena.copy_h2d_sync(runtime.h2d_stream(),input.data(),x,n*4);
  synGraphHandle graph=nullptr; synRecipeHandle recipe=nullptr; check(synGraphCreate(&graph,synDeviceGaudi2));
  std::array<synTensor,3> tensors{}; std::array<synSectionHandle,3> sections{};
  std::array<synLaunchTensorInfo,3> launch{};
  const char* names[]={"input","high","low"}; const uint64_t addresses[]={x,hi,lo};
  for(unsigned i=0;i<3;++i) {
    check(synSectionCreate(&sections[i],0,graph)); check(synSectionSetPersistent(sections[i],true));
    synTensorDescriptor d{};d.m_name=names[i];d.m_dataType=i?syn_type_bf16:syn_type_float;d.m_dims=2;
    d.m_sizes[0]=d.m_minSizes[0]=width;d.m_sizes[1]=d.m_minSizes[1]=rows;
    check(synTensorCreate(&tensors[i],&d,sections[i],0));
    launch[i].tensorName=names[i];launch[i].tensorType=DATA_TENSOR;launch[i].pTensorAddress=addresses[i];
    launch[i].tensorSize[0]=width;launch[i].tensorSize[1]=rows;
  }
  check(synNodeCreate(graph,&tensors[0],&tensors[1],1,2,nullptr,0,"gk_dense_activation_split_v1","split",nullptr,nullptr));
  check(synGraphCompile(&recipe,graph,"dense_activation_split_test",nullptr));
  uint64_t ws=0;check(synWorkspaceGetSize(&ws,recipe));initialize_launch_tensor_ids(recipe,launch.data(),launch.size());
  check(synGraphDestroy(graph));for(auto section:sections)check(synSectionDestroy(section));
  PreparedLaunch result{recipe,arena.ensure_workspace(ws),{launch.begin(),launch.end()},0};
  result.execute(runtime.compute_stream());runtime.synchronize_compute();
  std::vector<uint16_t> actual_hi(n),actual_lo(n);
  arena.copy_d2h_sync(runtime.d2h_stream(),hi,actual_hi.data(),n*2);
  arena.copy_d2h_sync(runtime.d2h_stream(),lo,actual_lo.data(),n*2);
  size_t bad_hi=0,bad_lo=0,nonzero=0;
  for(size_t i=0;i<n;++i) {
    const auto expected_hi=round_bf16(input[i]);
    const auto expected_lo=round_bf16(input[i]-widen(expected_hi));
    bad_hi+=actual_hi[i]!=expected_hi;bad_lo+=actual_lo[i]!=expected_lo;
    nonzero+=(actual_lo[i]&0x7fff)!=0;
  }
  std::cout<<"width="<<width<<" rows="<<rows<<" count="<<n<<" bad_high="<<bad_hi
           <<" bad_low="<<bad_lo<<" nonzero_low="<<nonzero<<"\n";
  check(synRecipeDestroy(recipe));if(bad_hi||bad_lo||!nonzero)throw std::runtime_error("split mismatch");
}
}
int main() {
  try {
    DeviceRuntime::Config cfg{};cfg.module_id=0;DeviceRuntime runtime(cfg);
    HbmArena arena(runtime.device_id(),512ULL*1024*1024);
    run(runtime,arena,6144,512);run(runtime,arena,2048,2048);
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
