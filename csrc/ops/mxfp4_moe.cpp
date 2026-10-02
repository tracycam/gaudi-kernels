#include "mxfp4_moe.hpp"
#include <cstdint>
#include <stdexcept>
#include <initializer_list>
namespace gaudi_kernels::mxfp4_moe {
static void ck(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static void require(synTensor t,synDataType dtype,std::initializer_list<uint64_t> sizes){
 if(!t)throw std::invalid_argument("missing routed down tensor");synTensorGeometry g{};synDataType d;ck(synTensorGetGeometry(t,&g,synGeometryMaxSizes),"shape");ck(synTensorGetDeviceDataType(t,&d),"dtype");if(g.dims!=sizes.size()||d!=dtype)throw std::invalid_argument("routed down tensor type/rank");unsigned i=0;for(auto s:sizes)if(g.sizes[i++]!=s)throw std::invalid_argument("routed down tensor size");
}
Resources append_routed_down(synGraphHandle graph,synTensor w,synTensor s,synTensor a,synTensor ids,synTensor routing,synTensor lut,synTensor y,int n,int k,int t,int r,int e,const Plan&p,const std::string&prefix){
 if(p.engine!=Engine::Direct512&&!p.allow_unqualified_experiments)throw std::invalid_argument("fusion candidate is not production qualified");
 if(!graph||n<1||k<1||t<1||r<1||e<1||n%512||k%32||!p.caller_certifies_fast_arithmetic)throw std::invalid_argument("aligned dimensions and explicit fast arithmetic certificate required");
 uint64_t nb=n/512,rows=uint64_t(t)*r;if(nb*k*e>2147483647u||nb*rows>2147483647u||uint64_t(n)*rows>2147483647u)throw std::invalid_argument("int32 routed down capacity exceeded");
 require(w,syn_type_packed_mxfp4,{256,nb*k*e});require(s,syn_type_uint8,{512,nb*(k/32)*e});require(a,syn_type_bf16,{uint64_t(k),rows});require(ids,syn_type_int32,{uint64_t(r),uint64_t(t)});require(routing,syn_type_single,{uint64_t(r),uint64_t(t)});require(lut,syn_type_bf16,{512,1});require(y,syn_type_single,{uint64_t(n),uint64_t(t)});
 bool narrow=p.engine==Engine::Fused256||p.engine==Engine::Fused256C;uint64_t taskblocks=narrow?nb*2:nb;
 uint32_t params[]={uint32_t(taskblocks),uint32_t(((uint64_t(1)<<32)+taskblocks-1)/taskblocks),uint32_t(e),uint32_t(((uint64_t(1)<<32)+r-1)/r)};Resources out;
 synTensor target=y;if(p.engine==Engine::Direct512){synTensorDescriptor d{};auto name=prefix+"_route_partials";d.m_name=name.c_str();d.m_dataType=syn_type_single;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=n;d.m_sizes[1]=d.m_minSizes[1]=rows;ck(synTensorCreate(&target,&d,nullptr,0),"route partial tensor");out.tensors.push_back(target);}
 if(narrow){
  auto view=[&](synTensor original,synDataType dtype,uint64_t x,uint64_t y,const std::string&name){synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=dtype;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=x;d.m_sizes[1]=d.m_minSizes[1]=y;synTensor v;ck(synTensorCreate(&v,&d,nullptr,0),"narrow view");out.tensors.push_back(v);ck(synNodeCreate(graph,&original,&v,1,1,nullptr,0,"reshape",name.c_str(),nullptr,nullptr),"narrow view reshape");return v;};
  w=view(w,syn_type_packed_mxfp4,128,nb*k*e*2,prefix+"_weight_half_view");s=view(s,syn_type_uint8,256,nb*(k/32)*e*2,prefix+"_scale_half_view");
 }
 synTensor inputs[]={w,s,a,lut,ids,routing};ck(synNodeCreate(graph,inputs,&target,6,1,params,16,p.engine==Engine::Fused256C?"gk_mxfp4_moe_fused256c":narrow?"gk_mxfp4_moe_fused256":p.engine==Engine::Fused512?"gk_mxfp4_moe_fused512":"gk_mxfp4_moe_direct512",(prefix+"_down").c_str(),nullptr,nullptr),"down");out.nodes=1;
 if(p.engine==Engine::Direct512){synTensor inputs[]={target,routing,ids};ck(synNodeCreate(graph,inputs,&y,3,1,&e,4,"gk_mxfp4_moe_combine_f32",(prefix+"_combine").c_str(),nullptr,nullptr),"routing combine");++out.nodes;}
 return out;
}
}
