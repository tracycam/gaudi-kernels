// Dtype-preserving graph-owned reshape for MoE producer edges. In particular,
// logical Long stays Long: this operator never reinterprets it as an int32
// tensor. The caller's explicit int32 cast and kernel DATA_I32 gate remain.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor();auto shape=s[1].toIntVector();auto type=x.scalar_type();
 TORCH_CHECK(type==at::kBFloat16||type==at::kFloat||type==at::kInt||type==at::kLong,"MoE reshape supports BF16,FP32,Int,Long only");
 TORCH_CHECK(x.is_contiguous()&&!x.requires_grad()&&!shape.empty()&&shape.size()<=4,"MoE reshape requires contiguous inference tensors/rank1..4");
 int64_t size=1;for(auto n:shape){TORCH_CHECK(n>0&&n<=INT32_MAX&&size<=INT64_MAX/n,"MoE reshape shape overflow/empty");size*=n;}
 TORCH_CHECK(size==x.numel(),"MoE reshape element count mismatch");return {{type,shape}};
}
Tensor reshape(Tensor x,std::vector<int64_t>shape){
 at::Stack s{x,shape};meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU required; no CPU fallback");
 auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_moe_graph::reshape");return op.execute(s)[0];
}
}
TORCH_LIBRARY(gaudi_moe_graph,m){
 m.def("reshape(Tensor input, int[] shape) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_moe_graph::reshape","reshape",meta,
    [](const at::Stack&,size_t&bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_moe_graph,HPU,m){m.impl("reshape",reshape);}
TORCH_LIBRARY_IMPL(gaudi_moe_graph,Meta,m){m.impl("reshape",[](Tensor x,std::vector<int64_t>shape){auto v=meta({x,shape});return at::empty(v[0].shape,x.options());});}
