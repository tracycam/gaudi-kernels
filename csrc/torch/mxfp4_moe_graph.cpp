// Experimental public current-graph primitives. No acquisition/native launch.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <synapse_common_types.hpp>
#include <climits>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const Tensor&t,at::ScalarType dtype,int rank){
 TORCH_CHECK(t.scalar_type()==dtype&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad(),"MXFP4 MoE dtype/rank/contiguity contract");
 TORCH_CHECK(t.numel()<=INT32_MAX,"MXFP4 MoE tensor index overflow");for(auto n:t.sizes())TORCH_CHECK(n>0&&n<INT32_MAX,"MXFP4 MoE empty/oversize shape");
}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();for(const auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"MXFP4 MoE device mismatch");}
int64_t scalar(const at::Stack&s,int i,int64_t lo=0){auto v=s[i].toInt();TORCH_CHECK(v>=lo&&v<=INT32_MAX,"MXFP4 MoE scalar range");return v;}
void ids_check(Tensor ids){check(ids,at::kInt,2);TORCH_CHECK(ids.size(0)>=2&&ids.size(0)<=65536,"M1 retains literal historical path; require 2<=T<=65536");}
Meta count_meta(const at::Stack&s){auto ids=s[0].toTensor();ids_check(ids);auto e=scalar(s,1,1);TORCH_CHECK(ids.size(1)<=e,"R>E");return {{at::kInt,{1,e}},{at::kInt,{1,e}}};}
Meta plan_meta(const at::Stack&s){auto ids=s[0].toTensor(),c=s[1].toTensor();ids_check(ids);check(c,at::kInt,2);devices(s);auto mode=scalar(s,2),slots=scalar(s,3,1);TORCH_CHECK(c.size(0)==1&&ids.size(1)<=c.size(1)&&mode<=1,"route plan geometry");int64_t want=c.size(1),rows=want*ids.size(0);if(mode){want=0;rows=0;for(int64_t lo=1,hi=16;lo<=ids.size(0);lo=hi+1,hi*=2){auto n=std::min(c.size(1),ids.numel()/lo),cap=std::min(hi,ids.size(0));want+=n;rows+=cap*n;}}TORCH_CHECK(slots==want&&rows<=INT32_MAX,"incorrect static route capacity");return {{at::kInt,ids.sizes().vec()},{at::kInt,{1,slots}}};}
Meta gather_meta(const at::Stack&s){auto x=s[0].toTensor(),ids=s[1].toTensor(),c=s[2].toTensor(),map=s[3].toTensor();check(x,at::kBFloat16,2);ids_check(ids);check(c,at::kInt,2);check(map,at::kInt,2);devices(s);auto cap=scalar(s,4,1),slot=scalar(s,5),batch=scalar(s,6,1);TORCH_CHECK(x.size(1)==6144&&x.size(0)==ids.size(0)&&c.size(0)==1&&map.size(0)==1&&slot+batch<=map.size(1)&&cap<=ids.size(0),"local GP gather geometry");TORCH_CHECK(batch*cap*x.size(1)<=INT32_MAX,"local gather index overflow");return {{at::kBFloat16,{batch,cap,x.size(1)}}};}
Meta decode_meta(const at::Stack&s){auto w=s[0].toTensor(),sc=s[1].toTensor(),lut=s[2].toTensor(),map=s[3].toTensor();check(w,at::kByte,2);check(sc,at::kByte,2);check(lut,at::kBFloat16,2);check(map,at::kInt,2);devices(s);auto k=scalar(s,4,1),n=scalar(s,5,1),nt=scalar(s,6,1),slot=scalar(s,7),batch=scalar(s,8,1),nb=scalar(s,9),layout=scalar(s,10);TORCH_CHECK(layout==1,"explicit HistoricalN512 layout=1 required; native-v2 same shape is incompatible");TORCH_CHECK(((k==6144&&n==512)||(k==256&&n==6144))&&nt%512==0&&nb%512==0&&nb+nt<=n&&slot+batch<=map.size(1)&&map.size(0)==1,"historical decode shard/tile geometry");TORCH_CHECK(w.size(1)==256&&sc.size(1)==512&&w.size(0)%((n/512)*k)==0&&sc.size(0)*32==w.size(0)&&lut.size(0)==1&&lut.size(1)==512,"historical owner geometry");TORCH_CHECK(batch*k*nt*2<=16*1024*1024,"decoded fragment exceeds 16 MiB");return {{at::kBFloat16,{batch,k,nt}}};}
Meta mm_meta(const at::Stack&s){auto x=s[0].toTensor(),w=s[1].toTensor();check(x,at::kBFloat16,3);check(w,at::kBFloat16,3);devices(s);TORCH_CHECK(x.size(0)==w.size(0)&&x.size(2)==w.size(1),"batch MME geometry");return {{at::kFloat,{x.size(0),x.size(1),w.size(2)}}};}
Meta gate_meta(const at::Stack&s){auto p=s[0].toTensor(),map=s[1].toTensor(),c=s[2].toTensor();check(p,at::kFloat,3);check(map,at::kInt,2);check(c,at::kInt,2);devices(s);auto slot=scalar(s,3);TORCH_CHECK(p.size(2)==512&&map.size(0)==1&&c.size(0)==1&&slot+p.size(0)<=map.size(1),"expert-order GP gate geometry");return {{at::kBFloat16,{p.size(0),p.size(1),256}}};}
Meta combine_meta(const at::Stack&s){auto p=s[0].toTensor(),ids=s[1].toTensor(),r=s[2].toTensor(),inv=s[3].toTensor(),st=s[4].toTensor();check(p,at::kFloat,2);ids_check(ids);check(r,at::kFloat,2);check(inv,at::kInt,2);check(st,at::kInt,2);devices(s);auto e=scalar(s,5,1);TORCH_CHECK(ids.sizes()==r.sizes()&&ids.sizes()==inv.sizes()&&st.size(0)==1&&st.size(1)==e&&p.size(1)%512==0,"ordered combine geometry");return {{at::kFloat,{ids.size(0),p.size(1)}}};}
Meta reshape_meta(const at::Stack&s){auto x=s[0].toTensor();auto shape=s[1].toIntVector();TORCH_CHECK(x.is_contiguous()&&!x.requires_grad()&&shape.size()>=1&&shape.size()<=3,"graph reshape contiguous rank1..3");TORCH_CHECK(x.scalar_type()==at::kBFloat16||x.scalar_type()==at::kFloat,"graph reshape BF16/FP32 only");int64_t count=1;for(auto n:shape){TORCH_CHECK(n>0&&n<=INT32_MAX&&count<=INT32_MAX/n,"reshape overflow");count*=n;}TORCH_CHECK(count==x.numel(),"reshape size mismatch");return {{x.scalar_type(),shape}};}
using MetaFn=Meta(*)(const at::Stack&);
std::vector<Tensor> invoke(const char*name,MetaFn fn,const at::Stack&s){auto meta=fn(s);if(s[0].toTensor().device().type()==at::kMeta){std::vector<Tensor>out;for(auto m:meta)out.push_back(at::empty(m.shape,s[0].toTensor().options().dtype(m.dtype)));return out;}TORCH_CHECK(s[0].toTensor().device().type()==at::kHPU,"HPU required, no CPU fallback");auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return descriptor.execute(s);}
std::tuple<Tensor,Tensor> count(Tensor ids,int64_t e){auto v=invoke("gaudi_moe_reference::count",count_meta,{ids,e});return {v[0],v[1]};}
std::tuple<Tensor,Tensor> plan(Tensor ids,Tensor c,int64_t mode,int64_t slots){auto v=invoke("gaudi_moe_reference::plan",plan_meta,{ids,c,mode,slots});return {v[0],v[1]};}
Tensor gather(Tensor x,Tensor ids,Tensor c,Tensor map,int64_t cap,int64_t slot,int64_t batch){return invoke("gaudi_moe_reference::gather",gather_meta,{x,ids,c,map,cap,slot,batch})[0];}
Tensor decode(Tensor w,Tensor s,Tensor lut,Tensor map,int64_t k,int64_t n,int64_t nt,int64_t slot,int64_t batch,int64_t nb,int64_t layout){return invoke("gaudi_moe_reference::decode",decode_meta,{w,s,lut,map,k,n,nt,slot,batch,nb,layout})[0];}
Tensor mm(Tensor x,Tensor w){return invoke("gaudi_moe_reference::batch_mm",mm_meta,{x,w})[0];}
Tensor gate(Tensor p,Tensor map,Tensor c,int64_t slot){return invoke("gaudi_moe_reference::gate_rows",gate_meta,{p,map,c,slot})[0];}
Tensor combine(Tensor p,Tensor ids,Tensor r,Tensor inv,Tensor st,int64_t e){return invoke("gaudi_moe_reference::combine",combine_meta,{p,ids,r,inv,st,e})[0];}
Tensor reshape(Tensor x,std::vector<int64_t>shape){return invoke("gaudi_moe_reference::reshape",reshape_meta,{x,shape})[0];}
std::shared_ptr<void> nop(const at::Stack&,size_t&n){n=0;return nullptr;}
template<size_t N>std::shared_ptr<void> params(size_t&bytes,const std::array<int32_t,N>&values){bytes=N*4;return std::make_shared<std::array<int32_t,N>>(values);}
}
TORCH_LIBRARY(gaudi_moe_reference,m){using habana::custom_op::registerUserCustomOp;
 m.def("count(Tensor ids, int experts) -> (Tensor, Tensor)");m.def("plan(Tensor ids, Tensor counts, int mode, int slots) -> (Tensor, Tensor)");
 m.def("gather(Tensor x, Tensor ids, Tensor counts, Tensor expert_map, int capacity, int slot_begin, int expert_batch) -> Tensor");
 m.def("decode(Tensor packed, Tensor scales, Tensor lut, Tensor expert_map, int k, int n, int n_tile, int slot_begin, int expert_batch, int n_begin, int layout) -> Tensor");
 m.def("batch_mm(Tensor x, Tensor w) -> Tensor");m.def("gate_rows(Tensor partial, Tensor expert_map, Tensor counts, int slot_begin) -> Tensor");
 m.def("combine(Tensor partial, Tensor ids, Tensor routing, Tensor inverse, Tensor status, int experts) -> Tensor");m.def("reshape(Tensor input, int[] shape) -> Tensor");
 registerUserCustomOp("gaudi_moe_reference::count","gk_mxfp4_graph_count",count_meta,nop);
 registerUserCustomOp("gaudi_moe_reference::plan","gk_mxfp4_graph_plan",plan_meta,[](const at::Stack&s,size_t&n){return params<2>(n,{int32_t(s[1].toTensor().size(1)),int32_t(s[2].toInt())});});
 registerUserCustomOp("gaudi_moe_reference::gather","gk_mxfp4_graph_gather",gather_meta,[](const at::Stack&s,size_t&n){return params<2>(n,{int32_t(s[4].toInt()),int32_t(s[5].toInt())});});
 registerUserCustomOp("gaudi_moe_reference::decode","gk_mxfp4_graph_decode_historical",decode_meta,[](const at::Stack&s,size_t&n){return params<3>(n,{int32_t(s[5].toInt()/512),int32_t(s[7].toInt()),int32_t(s[9].toInt()/512)});});
 registerUserCustomOp("gaudi_moe_reference::batch_mm","batch_gemm",mm_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=sizeof(synGEMMParams);return std::make_shared<synGEMMParams>(false,false);});
 registerUserCustomOp("gaudi_moe_reference::gate_rows","gk_mxfp4_graph_gate_rows",gate_meta,[](const at::Stack&s,size_t&n){return params<1>(n,{int32_t(s[3].toInt())});});
 registerUserCustomOp("gaudi_moe_reference::combine","gk_mxfp4_graph_combine",combine_meta,[](const at::Stack&s,size_t&n){return params<1>(n,{int32_t(s[5].toInt())});});
 registerUserCustomOp("gaudi_moe_reference::reshape","reshape",reshape_meta,nop);
}
#define GK_MOE_IMPL m.impl("count",count);m.impl("plan",plan);m.impl("gather",gather);m.impl("decode",decode);m.impl("batch_mm",mm);m.impl("gate_rows",gate);m.impl("combine",combine);m.impl("reshape",reshape);
TORCH_LIBRARY_IMPL(gaudi_moe_reference,HPU,m){GK_MOE_IMPL}
TORCH_LIBRARY_IMPL(gaudi_moe_reference,Meta,m){GK_MOE_IMPL}
