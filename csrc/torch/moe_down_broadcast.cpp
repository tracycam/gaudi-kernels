// Separate public CustomOp for unchanged K32 down MAC with grouped activations.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using T=at::Tensor;
namespace {
habana::PartialOutputMetaDataVector meta(const at::Stack&s){
 auto x=s[2].toTensor(),ids=s[4].toTensor();
 TORCH_CHECK(x.dim()==2&&x.size(1)==256&&x.scalar_type()==at::kBFloat16,"down expects BF16[slots,256]");
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>0&&ids.size(1)>=1&&ids.size(1)<=384&&x.size(0)==ids.numel(),"down route/activation shape mismatch");
 return {{at::kFloat,{1,ids.numel()*6144}}};
}
T run(T w,T s,T x,T table,T ids){
 at::Stack stack{w,s,x,table,ids};meta(stack);
 auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_down_activation::broadcast");
 return op.execute(stack)[0];
}
}
TORCH_LIBRARY(gaudi_down_activation,m){
 m.def("broadcast(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_down_activation::broadcast","downact_direct_down_broadcast",meta,
   [](const at::Stack&,size_t&bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_down_activation,HPU,m){m.impl("broadcast",run);}
TORCH_LIBRARY_IMPL(gaudi_down_activation,Meta,m){m.impl("broadcast",[](T w,T s,T x,T table,T ids){return at::empty(meta({w,s,x,table,ids})[0].shape,x.options().dtype(at::kFloat));});}
