// Logical graph-owned edges for temporary query/K/V/context and slot metadata.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor();auto shape=s[1].toIntVector();
 TORCH_CHECK((x.scalar_type()==at::kBFloat16||x.scalar_type()==at::kInt||x.scalar_type()==at::kLong)&&x.is_contiguous()&&!x.requires_grad(),"SWA reshape requires contiguous inference BF16/I32/Long");
 TORCH_CHECK(!shape.empty()&&shape.size()<=4,"SWA reshape rank must be 1..4");int64_t count=1;
 for(auto n:shape){TORCH_CHECK(n>0&&n<=INT32_MAX&&count<=INT64_MAX/n,"SWA reshape shape overflow");count*=n;}
 TORCH_CHECK(count==x.numel(),"SWA reshape changes element count");return {{x.scalar_type(),shape}};
}
Tensor run(Tensor x,std::vector<int64_t> shape){at::Stack s{x,shape};meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_swa128::reshape");return d.execute(s)[0];}
}
TORCH_LIBRARY_FRAGMENT(gaudi_swa128,m){m.def("reshape(Tensor input, int[] shape) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::reshape","reshape",meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});}
TORCH_LIBRARY_IMPL(gaudi_swa128,HPU,m){m.impl("reshape",run);}
TORCH_LIBRARY_IMPL(gaudi_swa128,Meta,m){m.impl("reshape",[](Tensor x,std::vector<int64_t> shape){auto m=meta({x,shape});return at::empty(m[0].shape,x.options());});}
