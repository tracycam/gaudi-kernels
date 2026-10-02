#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using T=at::Tensor;
namespace {
habana::PartialOutputMetaDataVector meta(const at::Stack&s){
 auto x=s[2].toTensor(),ids=s[4].toTensor();
 TORCH_CHECK(ids.dim()==2&&ids.size(0)>0&&ids.size(1)>=1&&ids.size(1)<=384&&ids.numel()<=INT_MAX/1536,"GP scale-tail route domain");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(1)==6144&&x.size(0)==ids.size(0),"GP scale-tail BF16[M,6144] required");
 return {{at::kFloat,{1,ids.numel()*1536}}};
}
T run(T w,T s,T x,T table,T ids){at::Stack stack{w,s,x,table,ids};meta(stack);auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_gp_scale_tail::gp");return op.execute(stack)[0];}
}
TORCH_LIBRARY(gaudi_gp_scale_tail,m){m.def("gp(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_gp_scale_tail::gp","gk_moe_gp_scale_tail_v1",meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});}
TORCH_LIBRARY_IMPL(gaudi_gp_scale_tail,HPU,m){m.impl("gp",run);}
TORCH_LIBRARY_IMPL(gaudi_gp_scale_tail,Meta,m){m.impl("gp",[](T w,T s,T x,T t,T ids){return at::empty(meta({w,s,x,t,ids})[0].shape,x.options().dtype(at::kFloat));});}
