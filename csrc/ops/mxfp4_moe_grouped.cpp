#include "mxfp4_moe_grouped.hpp"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
namespace gaudi_kernels::mxfp4_moe_grouped {
static void ck(synStatus s,const char*what){if(s!=synSuccess)throw std::runtime_error(std::string(what)+": "+std::to_string(s));}
static void require(synTensor t,synDataType type,std::initializer_list<uint64_t> sizes){
 if(!t){throw std::invalid_argument("missing grouped MoE tensor");}
 synTensorGeometry g{};synDataType actual;ck(synTensorGetGeometry(t,&g,synGeometryMaxSizes),"shape");ck(synTensorGetDeviceDataType(t,&actual),"dtype");if(actual!=type||g.dims!=sizes.size())throw std::invalid_argument("grouped tensor dtype/rank");unsigned i=0;for(auto s:sizes)if(g.sizes[i++]!=s)throw std::invalid_argument("grouped tensor shape");
}
Budget budget(int n,int k,int t,int r,int e,const Plan&p){
 if(n<1||k<1||t<1||r<1||e<1||r>e||n%512||k%32||p.expert_batch<1||p.n_tile<512||p.n_tile%512||p.decoded_sram_limit>16ull*1024*1024||!p.unique_experts_per_token)throw std::invalid_argument("aligned dimensions, SRAM budget and unique top-k contract required");
 if(p.gate_up_epilogue&&p.n_tile<n)throw std::invalid_argument("GP epilogue prototype requires one full-N tile");
 Budget b;b.capacity=p.capacity?p.capacity:t;b.n_tile=std::min(n,p.n_tile);
 if(b.capacity<1||(b.capacity<t&&!p.allow_capacity_experiment))throw std::invalid_argument("capacity<T is an unsupported production route; explicit diagnostic only");
 b.expert_batch=std::min<uint64_t>(std::min(e,p.expert_batch),p.decoded_sram_limit/(2ull*b.n_tile*k));
 if(!b.expert_batch)throw std::invalid_argument("one decoded expert tile exceeds SRAM budget");
 if(uint64_t(n)*k*e/2>INT32_MAX||uint64_t(k)*t*r>INT32_MAX||uint64_t(n)*b.capacity*e>INT32_MAX)throw std::invalid_argument("grouped prototype int32 tensor capacity exceeded");
 b.decoded_sram_bytes=2ull*b.n_tile*k*b.expert_batch;b.grouped_activation_bytes=2ull*k*b.capacity*e;
 b.fp32_expert_outputs_upper_bytes=4ull*n*b.capacity*e;b.original_weight_scale_bytes=uint64_t(n)*k*e*17/32;
 b.mme_row_slots=uint64_t(b.capacity)*e;b.combine_nodes=(n+b.n_tile-1)/b.n_tile;
 b.mme_nodes=b.decode_nodes=b.combine_nodes*((e+b.expert_batch-1)/b.expert_batch);return b;
}
Resources append(synGraphHandle graph,synTensor w,synTensor s,synTensor a,synTensor ids,synTensor routing,synTensor lut,synTensor y,synTensor overflow,int n,int k,int t,int r,int e,const Plan&p,const std::string&prefix){
 if(!mxfp4_moe_layout::supported(p.weight_layout))throw std::invalid_argument("explicit HistoricalN512 or NativeN512V2 weight layout required");
 if(!graph||!p.scales_2_252||!p.caller_certifies_fast_arithmetic)throw std::invalid_argument("graph, scale eligibility and separate fast arithmetic certificate required");
 Resources out;out.budget=budget(n,k,t,r,e,p);const auto&b=out.budget;int nb=n/512;
 if(p.weight_layout==WeightLayout::HistoricalN512){require(w,syn_type_uint8,{256,uint64_t(k)*e*nb});require(s,syn_type_uint8,{512,uint64_t(k/32)*e*nb});}else{require(w,syn_type_uint8,{256,uint64_t(k),uint64_t(e)*nb});require(s,syn_type_uint8,{512,uint64_t(k/32),uint64_t(e)*nb});}require(a,syn_type_bf16,{uint64_t(k),uint64_t(t)*(p.activation_per_token?1:r)});require(ids,syn_type_int32,{uint64_t(r),uint64_t(t)});require(routing,syn_type_single,{uint64_t(r),uint64_t(t)});require(lut,syn_type_bf16,{512,1});require(y,p.gate_up_epilogue?syn_type_bf16:syn_type_single,{uint64_t(p.gate_up_epilogue?n/2:n),uint64_t(t)*(p.gate_up_epilogue?r:1)});require(overflow,syn_type_int32,{uint64_t(e),1});
 auto tensor=[&](const std::string&name,synDataType type,std::vector<int>shape,synSectionHandle section=nullptr){synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=shape.size();for(unsigned i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];synTensor result;ck(synTensorCreate(&result,&d,section,0),"grouped tensor");out.tensors.push_back(result);return result;};
 auto node=[&](std::vector<synTensor>inputs,std::vector<synTensor>outputs,const void*params,unsigned bytes,const char*guid,const std::string&name){synNodeId id;ck(synNodeCreateWithId(graph,inputs.data(),outputs.data(),inputs.size(),outputs.size(),params,bytes,guid,name.c_str(),&id,nullptr,nullptr),"grouped node");return id;};
 auto grouped_a=tensor(prefix+"_A",syn_type_bf16,{k,b.capacity,e}),inverse=tensor(prefix+"_inverse",syn_type_int32,{r,t}),counts=tensor(prefix+"_counts",syn_type_int32,{e,1});
 int gp[]={b.capacity,e,int(p.activation_per_token)};node({a,ids},{grouped_a,inverse,counts,overflow},gp,12,"gk_mxfp4_group_prepare",prefix+"_gather");
 std::vector<synTensor>aviews;std::vector<int>sizes;
 for(int start=0;start<e;start+=b.expert_batch){int width=std::min(e-start,b.expert_batch);sizes.push_back(width);aviews.push_back(width==e?grouped_a:tensor(prefix+"_A"+std::to_string(start),syn_type_bf16,{k,b.capacity,width}));}
 synSplitParams split{2};if(aviews.size()>1)node({grouped_a},aviews,&split,sizeof(split),"split",prefix+"_A_views");
 synSectionHandle scratch;ck(synSectionCreate(&scratch,0,graph),"scratch section");ck(synSectionSetPersistent(scratch,false),"nonpersistent");ck(synSectionSetRMW(scratch,true),"SRAM request");out.sections.push_back(scratch);
 synNodeId previous=0;std::vector<synTensor>token_tiles;
 for(int n0=0;n0<n;n0+=b.n_tile){int width=std::min(b.n_tile,n-n0);std::vector<synTensor>expert_tiles;
  for(unsigned batch=0;batch<sizes.size();++batch){int e0=batch*b.expert_batch,eb=sizes[batch];auto tag=prefix+"_n"+std::to_string(n0)+"_e"+std::to_string(e0);
   auto decoded=tensor(tag+"_decoded_sram",syn_type_bf16,{width,k,eb},scratch),result=tensor(tag+"_f32",syn_type_single,{width,b.capacity,eb});
   int dp[]={nb,e0,n0/512,b.capacity};auto decoder=node({w,s,lut,counts},{decoded},dp,16,p.weight_layout==WeightLayout::HistoricalN512?"gk_mxfp4_group_decode_legacy_n512":"gk_mxfp4_group_decode",tag+"_decode");
   if(previous)ck(synNodeDependencySet(graph,&previous,&decoder,1,1),"scratch reuse");
   synGEMMParams gemm{false,false};previous=node({aviews[batch],decoded},{result},&gemm,sizeof(gemm),"batch_gemm",tag+"_batch_mme");expert_tiles.push_back(result);
  }
  synTensor full=expert_tiles[0];if(expert_tiles.size()>1){full=tensor(prefix+"_all_experts_"+std::to_string(n0),syn_type_single,{width,b.capacity,e});synConcatenateParams cp{2};node(expert_tiles,{full},&cp,sizeof(cp),"concat",prefix+"_experts_view_"+std::to_string(n0));}
  auto tile=tensor(prefix+"_tokens_"+std::to_string(n0),p.gate_up_epilogue?syn_type_bf16:syn_type_single,{p.gate_up_epilogue?width/2:width,p.gate_up_epilogue?t*r:t});node({full,ids,routing,inverse,overflow},{tile},gp,8,p.gate_up_epilogue?"gk_mxfp4_group_gate":"gk_mxfp4_group_combine",prefix+"_route_combine_"+std::to_string(n0));token_tiles.push_back(tile);
 }
 if(token_tiles.size()>1){synConcatenateParams cp{0};node(token_tiles,{y},&cp,sizeof(cp),"concat",prefix+"_output_view");}else node(token_tiles,{y},nullptr,0,"reshape",prefix+"_output_view");
 return out;
}
}
