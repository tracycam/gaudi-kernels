// Logical split in the current Synapse graph, with an explicit producer edge.
// Torch Lazy strided views lose this edge through the public CustomOp bridge.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <synapse_common_types.hpp>
#include <climits>
#include <array>
using Tensor=at::Tensor;
using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const Tensor&t,at::ScalarType dtype,int rank){
 TORCH_CHECK(t.scalar_type()==dtype&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad(),"MXFP4 MoE dtype/rank/contiguity contract");
 TORCH_CHECK(t.numel()<=INT32_MAX,"MXFP4 MoE tensor index overflow");for(auto n:t.sizes())TORCH_CHECK(n>0&&n<INT32_MAX,"MXFP4 MoE empty/oversize shape");
}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();for(const auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"MXFP4 MoE device mismatch");}
int64_t scalar(const at::Stack&s,int i,int64_t lo=0){auto v=s[i].toInt();TORCH_CHECK(v>=lo&&v<=INT32_MAX,"MXFP4 MoE scalar range");return v;}
Meta decode_meta(const at::Stack&s){auto w=s[0].toTensor(),sc=s[1].toTensor(),lut=s[2].toTensor(),map=s[3].toTensor();check(w,at::kByte,2);check(sc,at::kByte,2);check(lut,at::kBFloat16,2);check(map,at::kInt,2);devices(s);auto k=scalar(s,4,1),n=scalar(s,5,1),nt=scalar(s,6,1),slot=scalar(s,7),batch=scalar(s,8,1),nb=scalar(s,9),layout=scalar(s,10);TORCH_CHECK(layout==1,"explicit HistoricalN512 layout=1 required; native-v2 same shape is incompatible");TORCH_CHECK(((k==6144&&n==512)||(k==256&&n==6144))&&nt%512==0&&nb%512==0&&nb+nt<=n&&slot+batch<=map.size(1)&&map.size(0)==1,"historical decode shard/tile geometry");TORCH_CHECK(w.size(1)==256&&sc.size(1)==512&&w.size(0)%((n/512)*k)==0&&sc.size(0)*32==w.size(0)&&lut.size(0)==1&&lut.size(1)==512,"historical owner geometry");TORCH_CHECK(batch*k*nt*2<=16*1024*1024,"decoded fragment exceeds 16 MiB");return {{at::kBFloat16,{batch,k,nt}}};}
Tensor decode(Tensor w,Tensor sc,Tensor lut,Tensor map,int64_t k,int64_t n,int64_t nt,int64_t slot,int64_t batch,int64_t nb,int64_t layout){
 at::Stack stack{w,sc,lut,map,k,n,nt,slot,batch,nb,layout};auto m=decode_meta(stack);
 TORCH_CHECK(batch<=2,"stream decoder permits at most two experts per fragment");
 if(w.device().type()==at::kMeta)return at::empty(m[0].shape,w.options().dtype(at::kBFloat16));
 TORCH_CHECK(w.device().type()==at::kHPU,"HPU required");
 auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_moe_stream::decode");return op.execute(stack)[0];
}
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor();auto chunk=s[1].toInt();
 TORCH_CHECK(x.dim()==3&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&!x.requires_grad(),"stream split requires contiguous BF16 rank3");
 TORCH_CHECK(x.size(0)>0&&x.size(0)<=64&&x.size(1)>0&&x.size(1)<=8&&x.size(2)==6144&&chunk>=1&&chunk<=2,"bounded small-M stream split");
 Meta out;for(int64_t begin=0;begin<x.size(0);begin+=chunk)out.push_back({at::kBFloat16,{std::min(chunk,x.size(0)-begin),x.size(1),x.size(2)}});
 return out;
}
std::vector<Tensor> split(Tensor x,int64_t chunk){
 at::Stack s{x,chunk};auto m=meta(s);
 if(x.device().type()==at::kMeta){std::vector<Tensor> out;for(auto p:m)out.push_back(at::empty(p.shape,x.options()));return out;}
 TORCH_CHECK(x.device().type()==at::kHPU,"HPU required, no CPU fallback");
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_moe_stream::split");
 return descriptor.execute(s);
}
}
TORCH_LIBRARY(gaudi_moe_stream,m){
 m.def("decode(Tensor packed, Tensor scales, Tensor lut, Tensor expert_map, int k, int n, int n_tile, int slot_begin, int expert_batch, int n_begin, int layout) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_moe_stream::decode","gk_moe_stream_decode_historical_k8",decode_meta,
  [](const at::Stack&s,size_t&bytes)->std::shared_ptr<void>{bytes=12;return std::make_shared<std::array<int32_t,3>>(std::array<int32_t,3>{int32_t(s[5].toInt()/512),int32_t(s[7].toInt()),int32_t(s[9].toInt()/512)});});
 m.def("split(Tensor input, int chunk) -> Tensor[]");
 habana::custom_op::registerUserCustomOp("gaudi_moe_stream::split","split",meta,
  [](const at::Stack&,size_t&bytes)->std::shared_ptr<void>{bytes=sizeof(synSplitParams);return std::make_shared<synSplitParams>(synSplitParams{2});});
}
TORCH_LIBRARY_IMPL(gaudi_moe_stream,HPU,m){m.impl("split",split);m.impl("decode",decode);}
TORCH_LIBRARY_IMPL(gaudi_moe_stream,Meta,m){m.impl("split",split);m.impl("decode",decode);}
