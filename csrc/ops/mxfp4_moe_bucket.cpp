#include "mxfp4_moe_bucket.hpp"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
namespace gaudi_kernels::mxfp4_moe_bucket {
static void ck(synStatus s,const char*what){if(s!=synSuccess)throw std::runtime_error(std::string(what)+": "+std::to_string(s));}
static void require(synTensor t,synDataType type,std::initializer_list<uint64_t> sizes){
 if(!t)throw std::invalid_argument("missing bucket tensor");
 synTensorGeometry g{};synDataType actual;ck(synTensorGetGeometry(t,&g,synGeometryMaxSizes),"shape");ck(synTensorGetDeviceDataType(t,&actual),"dtype");
 if(actual!=type||g.dims!=sizes.size())throw std::invalid_argument("bucket tensor dtype/rank");
 unsigned i=0;for(auto s:sizes)if(g.sizes[i++]!=s)throw std::invalid_argument("bucket tensor shape");
}
Budget budget(int n,int k,int t,int r,int e,const Plan&p){
 if(n<1||k<1||t<1||t>65536||r<1||e<1||r>e||n%512||k%32||p.expert_batch<1||p.n_tile<512||p.n_tile%512||p.decoded_sram_limit>16ull*1024*1024||(p.scratch_buffers!=1&&p.scratch_buffers!=2)||!p.unique_experts_per_token)
  throw std::invalid_argument("aligned dimensions, T<=65536, SRAM and unique top-k contract required");
 if(p.gate_up_epilogue&&p.n_tile<n)throw std::invalid_argument("GP prototype requires full-N tile");
 if(uint64_t(t)*r>INT32_MAX||uint64_t(n)*k/2>uint64_t(INT32_MAX)/e||uint64_t(k)*t*r>INT32_MAX)throw std::invalid_argument("int32 input capacity exceeded");
 uint64_t per_buffer_limit=p.decoded_sram_limit/p.scratch_buffers;
 if(p.scratch_buffers==2)per_buffer_limit=std::min<uint64_t>(per_buffer_limit,8ull*1024*1024);
 Budget b;b.n_tile=std::min(n,p.n_tile);b.expert_batch=std::min<uint64_t>(p.expert_batch,per_buffer_limit/(2ull*b.n_tile*k));
 if(!b.expert_batch)throw std::invalid_argument("decoded tile exceeds SRAM limit");
 uint64_t slots=0,rows=0;
 for(int lower=1,limit=16;lower<=t;lower=limit+1,limit*=2){
  int upper=std::min(t,limit),count=std::min<uint64_t>(e,uint64_t(t)*r/lower);
  if(slots>INT32_MAX||rows>INT32_MAX)throw std::invalid_argument("bucket prefix exceeds int32");
  b.buckets.push_back({lower,upper,count,int(slots),int(rows)});slots+=count;rows+=uint64_t(count)*upper;
  b.mme_nodes+=(count+b.expert_batch-1)/b.expert_batch;
 }
 if(slots>INT32_MAX||rows>INT32_MAX||uint64_t(k)*rows>INT32_MAX||uint64_t(n)*rows>INT32_MAX)throw std::invalid_argument("int32 grouped capacity exceeded");
 b.expert_slots=slots;b.row_slots=rows;b.mme_nodes*=(n+b.n_tile-1)/b.n_tile;b.decode_nodes=b.mme_nodes;
 b.grouped_activation_bytes=2ull*k*rows;b.fp32_output_bytes=4ull*n*rows;b.per_buffer_sram_bytes=2ull*b.n_tile*k*b.expert_batch;b.decoded_sram_bytes=b.per_buffer_sram_bytes*p.scratch_buffers;return b;
}
Resources append(synGraphHandle graph,synTensor w,synTensor s,synTensor a,synTensor ids,synTensor routing,synTensor lut,synTensor y,synTensor status,int n,int k,int t,int r,int e,const Plan&p,const std::string&prefix){
 if(!mxfp4_moe_layout::supported(p.weight_layout))throw std::invalid_argument("explicit HistoricalN512 or NativeN512V2 weight layout required");
 if(!graph||!p.scales_2_252||!p.caller_certifies_fast_arithmetic)throw std::invalid_argument("graph and separate fast arithmetic certificate required");
 Resources out;out.budget=budget(n,k,t,r,e,p);const auto&b=out.budget;int nb=n/512;
 if(p.weight_layout==WeightLayout::HistoricalN512){require(w,syn_type_uint8,{256,uint64_t(k)*e*nb});require(s,syn_type_uint8,{512,uint64_t(k/32)*e*nb});}else{require(w,syn_type_uint8,{256,uint64_t(k),uint64_t(e)*nb});require(s,syn_type_uint8,{512,uint64_t(k/32),uint64_t(e)*nb});}require(a,syn_type_bf16,{uint64_t(k),uint64_t(t)*(p.activation_per_token?1:r)});require(ids,syn_type_int32,{uint64_t(r),uint64_t(t)});require(routing,syn_type_single,{uint64_t(r),uint64_t(t)});require(lut,syn_type_bf16,{512,1});require(status,syn_type_int32,{uint64_t(e),1});require(y,p.gate_up_epilogue?syn_type_bf16:syn_type_single,{uint64_t(p.gate_up_epilogue?n/2:n),uint64_t(t)*(p.gate_up_epilogue?r:1)});
 auto tensor=[&](const std::string&name,synDataType type,std::vector<int>shape,synSectionHandle section=nullptr){synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=shape.size();for(unsigned i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];synTensor result;ck(synTensorCreate(&result,&d,section,0),"bucket tensor");out.tensors.push_back(result);return result;};
 auto node=[&](std::vector<synTensor>in,std::vector<synTensor>outputs,const void*params,unsigned bytes,const char*guid,const std::string&name){synNodeId id;ck(synNodeCreateWithId(graph,in.data(),outputs.data(),in.size(),outputs.size(),params,bytes,guid,name.c_str(),&id,nullptr,nullptr),"bucket node");return id;};
 auto counts=tensor(prefix+"_counts",syn_type_int32,{e,1}),map=tensor(prefix+"_expert_map",syn_type_int32,{b.expert_slots,1}),inverse=tensor(prefix+"_inverse",syn_type_int32,{r,t}),grouped=tensor(prefix+"_A",syn_type_bf16,{k,b.row_slots});
 node({ids},{counts,status},nullptr,0,"gk_mxfp4_bucket_count",prefix+"_counts");int gather_params[]={e,int(p.activation_per_token)};
 node({a,ids,counts},{grouped,inverse,map},gather_params,sizeof(gather_params),"gk_mxfp4_bucket_gather",prefix+"_gather");
 struct Fragment {int capacity,slots,slot_base; synTensor a;};std::vector<Fragment>fragments;std::vector<synTensor>flat_views;
 for(auto bucket:b.buckets)for(int start=0;start<bucket.expert_slots;start+=b.expert_batch){
  int size=std::min(b.expert_batch,bucket.expert_slots-start);auto tag=prefix+"_slot"+std::to_string(bucket.slot_base+start);
  auto flat=tensor(tag+"_A_flat",syn_type_bf16,{k,bucket.upper*size});flat_views.push_back(flat);
  auto view=tensor(tag+"_A_view",syn_type_bf16,{k,bucket.upper,size});node({flat},{view},nullptr,0,"reshape",tag+"_reshape_A");fragments.push_back({bucket.upper,size,bucket.slot_base+start,view});
 }
 if(flat_views.size()>1){synSplitParams split{1};node({grouped},flat_views,&split,sizeof(split),"split",prefix+"_A_views");}else node({grouped},flat_views,nullptr,0,"reshape",prefix+"_A_view");
 std::vector<synSectionHandle>scratch(p.scratch_buffers);for(auto&section:scratch){ck(synSectionCreate(&section,0,graph),"scratch");ck(synSectionSetPersistent(section,false),"nonpersistent");ck(synSectionSetRMW(section,true),"SRAM request");out.sections.push_back(section);}
 std::vector<synNodeId>previous(p.scratch_buffers,0);unsigned ordinal=0;std::vector<synTensor>token_tiles;
 for(int n0=0;n0<n;n0+=b.n_tile){int width=std::min(b.n_tile,n-n0);std::vector<synTensor>results;
  for(auto f:fragments){auto tag=prefix+"_n"+std::to_string(n0)+"_slot"+std::to_string(f.slot_base);
   unsigned slot=ordinal++%p.scratch_buffers;
   auto decoded=tensor(tag+"_decoded_sram",syn_type_bf16,{width,k,f.slots},scratch[slot]),result=tensor(tag+"_f32",syn_type_single,{width,f.capacity,f.slots});int params[]={nb,f.slot_base,n0/512};
   auto decoder=node({w,s,lut,map},{decoded},params,sizeof(params),p.weight_layout==WeightLayout::HistoricalN512?"gk_mxfp4_bucket_decode_legacy_n512":"gk_mxfp4_bucket_decode",tag+"_decode");if(previous[slot])ck(synNodeDependencySet(graph,&previous[slot],&decoder,1,1),"scratch reuse");synGEMMParams gemm{false,false};previous[slot]=node({f.a,decoded},{result},&gemm,sizeof(gemm),"batch_gemm",tag+"_batch_mme");
   auto flat=tensor(tag+"_flat",syn_type_single,{width,f.capacity*f.slots});node({result},{flat},nullptr,0,"reshape",tag+"_flat_result");results.push_back(flat);
  }
  synTensor full=results[0];if(results.size()>1){full=tensor(prefix+"_all_rows_"+std::to_string(n0),syn_type_single,{width,b.row_slots});synConcatenateParams cp{1};node(results,{full},&cp,sizeof(cp),"concat",prefix+"_row_view_"+std::to_string(n0));}
  auto tile=tensor(prefix+"_tokens_"+std::to_string(n0),p.gate_up_epilogue?syn_type_bf16:syn_type_single,{p.gate_up_epilogue?width/2:width,p.gate_up_epilogue?t*r:t});node({full,ids,routing,inverse,status},{tile},&e,sizeof(e),p.gate_up_epilogue?"gk_mxfp4_bucket_gate":"gk_mxfp4_bucket_combine",prefix+"_epilogue_"+std::to_string(n0));token_tiles.push_back(tile);
 }
 if(token_tiles.size()>1){synConcatenateParams cp{0};node(token_tiles,{y},&cp,sizeof(cp),"concat",prefix+"_output_view");}else node(token_tiles,{y},nullptr,0,"reshape",prefix+"_output_view");return out;
}
}
