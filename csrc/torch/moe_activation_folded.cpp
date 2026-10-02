// Distinct public descriptors; no replacement of qualified GP/down operators.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta metadata(const at::Stack&s,bool down){
 auto x=s[2].toTensor(),ids=s[4].toTensor();
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>0&&ids.size(1)>=1&&ids.size(1)<=384&&ids.numel()<=INT_MAX/(down?6144:1536),"folded route shape/domain");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(1)==(down?256:6144)&&x.size(0)==(down?ids.numel():ids.size(0)),"folded activation shape/type");
 return {{at::kFloat,{1,ids.numel()*(down?6144:1536)}}};
}
T call(T w,T s,T x,T table,T ids,bool down){at::Stack stack{w,s,x,table,ids};metadata(stack,down);auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(down?"gaudi_activation_folded::down":"gaudi_activation_folded::gp");return op.execute(stack)[0];}
}
TORCH_LIBRARY(gaudi_activation_folded,m){
 m.def("gp(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
 m.def("down(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_activation_folded::gp","gk_moe_gp_folded_v1",[](const at::Stack&s){return metadata(s,false);},[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 habana::custom_op::registerUserCustomOp("gaudi_activation_folded::down","gk_moe_down_folded_v1",[](const at::Stack&s){return metadata(s,true);},[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_activation_folded,HPU,m){
 m.impl("gp",[](T w,T s,T x,T t,T ids){return call(w,s,x,t,ids,false);});
 m.impl("down",[](T w,T s,T x,T t,T ids){return call(w,s,x,t,ids,true);});
}
TORCH_LIBRARY_IMPL(gaudi_activation_folded,Meta,m){
 m.impl("gp",[](T w,T s,T x,T t,T ids){return at::empty(metadata({w,s,x,t,ids},false)[0].shape,x.options().dtype(at::kFloat));});
 m.impl("down",[](T w,T s,T x,T t,T ids){return at::empty(metadata({w,s,x,t,ids},true)[0].shape,x.options().dtype(at::kFloat));});
}
