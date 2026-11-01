// SPDX-License-Identifier: Apache-2.0
// Verify the actual TPC gather against a CPU page-table reference, including
// the first window, unaligned chunks, and permuted physical page numbers.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
using namespace gaudirt::runtime;
namespace {
void check(synStatus s){if(s!=synSuccess)throw std::runtime_error("Synapse status " + std::to_string(s));}
struct Tensor { std::string name; synTensor tensor=nullptr; synSectionHandle section=nullptr; synLaunchTensorInfo binding{}; size_t bytes=0; };
}
int main(){
 DeviceRuntime runtime(DeviceRuntime::Config{}); HbmArena arena(runtime.device_id(),64ULL*1024*1024);
 synGraphHandle graph=nullptr;synRecipeHandle recipe=nullptr;check(synGraphCreate(&graph,synDeviceGaudi2));
 std::vector<Tensor> t;t.reserve(7);
 auto make=[&](const char* name,synDataType type,std::initializer_list<uint64_t> shape,size_t element){
  t.emplace_back();auto& x=t.back();x.name=name;x.bytes=element;
  check(synSectionCreate(&x.section,0,graph));check(synSectionSetPersistent(x.section,true));
  synTensorDescriptor d{};d.m_name=x.name.c_str();d.m_dataType=type;d.m_dims=shape.size();unsigned i=0;
  for(auto n:shape){d.m_sizes[i]=d.m_minSizes[i]=n;x.binding.tensorSize[i++]=n;x.bytes*=n;}
  check(synTensorCreate(&x.tensor,&d,x.section,0));x.binding.tensorName=x.name.c_str();x.binding.tensorType=DATA_TENSOR;
  return t.size()-1;
 };
 constexpr int pages=64,slots=pages*128,rows=512;
 const auto key=make("key",syn_type_bf16,{192,1,slots},2), value=make("value",syn_type_bf16,{128,1,slots},2);
 const auto table=make("pages",syn_type_int32,{rows*2},4), groups=make("groups",syn_type_int32,{rows*2},4);
 const auto positions=make("positions",syn_type_int32,{rows},4);
 const auto outk=make("gathered_key",syn_type_bf16,{192,384,4},2), outv=make("gathered_value",syn_type_bf16,{128,384,4},2);
 synTensor in[]={t[key].tensor,t[value].tensor,t[table].tensor,t[groups].tensor,t[positions].tensor};
 synTensor out[]={t[outk].tensor,t[outv].tensor};
 check(synNodeCreate(graph,in,out,5,2,nullptr,0,"gk_swa_mme_gather_v1","gather",nullptr,nullptr));
 check(synGraphCompile(&recipe,graph,"swa_mme_pages_test",nullptr));uint64_t workspace=0;check(synWorkspaceGetSize(&workspace,recipe));
 std::vector<synLaunchTensorInfo> bindings;for(auto& x:t){x.binding.pTensorAddress=arena.allocate(x.bytes);bindings.push_back(x.binding);}
 initialize_launch_tensor_ids(recipe,bindings.data(),bindings.size());check(synGraphDestroy(graph));for(auto& x:t)check(synSectionDestroy(x.section));
 PreparedLaunch launch{recipe,arena.ensure_workspace(workspace),bindings,0};launch.validate();
 std::vector<uint16_t> k(slots*192),v(slots*128),gotk(384*4*192),gotv(384*4*128);
 for(size_t i=0;i<k.size();++i)k[i]=static_cast<uint16_t>((i*17+5)&65535);
 for(size_t i=0;i<v.size();++i)v[i]=static_cast<uint16_t>((i*31+7)&65535);
 auto put=[&](size_t i,const void* p){arena.copy_h2d_sync(runtime.h2d_stream(),p,bindings[i].pTensorAddress,t[i].bytes);};
 put(key,k.data());put(value,v.data());
 for(int start:{0,32,96,128,160,4096,4128,7168}){
  std::array<int32_t,rows> pos{};std::array<int32_t,rows*2> page{},group{};
  auto physical=[](int logical){return (logical*13+7)%pages;};
  for(int r=0;r<rows;++r){pos[r]=start+r;int first=std::max(0,pos[r]-127)/128,last=pos[r]/128;
   page[2*r]=physical(first);group[2*r]=r;page[2*r+1]=first==last?-1:physical(last);group[2*r+1]=first==last?-1:r;}
  put(table,page.data());put(groups,group.data());put(positions,pos.data());launch.execute(runtime.compute_stream());runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(),bindings[outk].pTensorAddress,gotk.data(),t[outk].bytes);
  arena.copy_d2h_sync(runtime.d2h_stream(),bindings[outv].pTensorAddress,gotv.data(),t[outv].bytes);
  for(int b=0;b<4;++b){int first=std::max(0,start+b*128-127)/128,last=(start+b*128+127)/128;
   for(int j=0;j<384;++j){int logical=first+j/128,slot=physical(logical)*128+j%128;bool valid=logical<=last;
    for(int i=0;i<192;++i)if(gotk[(b*384+j)*192+i]!=(valid?k[slot*192+i]:0))throw std::runtime_error("key page mismatch");
    for(int i=0;i<128;++i)if(gotv[(b*384+j)*128+i]!=(valid?v[slot*128+i]:0))throw std::runtime_error("value page mismatch");
   }
  }
  std::printf("start=%d permuted pages and gathered KV bitwise correct\n",start);
 }
 check(synRecipeDestroy(recipe));
}
