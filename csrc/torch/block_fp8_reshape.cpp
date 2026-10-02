// Graph-owned logical reshape. A temporary Torch view can become an expired
// lazy hpu::input at capture_end; this node keeps the producer/consumer edge.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using Tensor=at::Tensor;
using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor();auto shape=s[1].toIntVector();
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&!x.requires_grad(),"block reshape requires contiguous inference BF16");
 TORCH_CHECK(!shape.empty()&&shape.size()<=4,"block reshape supports rank1..4");
 int64_t count=1;
 for(auto n:shape){TORCH_CHECK(n>0&&n<=INT32_MAX&&count<=INT64_MAX/n,"block reshape dimension/product out of range");count*=n;}
 TORCH_CHECK(count==x.numel(),"block reshape element count mismatch");
 return {{at::kBFloat16,shape}};
}
Tensor reshape(Tensor x,std::vector<int64_t> shape){
 at::Stack s{x,shape};meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU required; no reshape CPU fallback");
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_block_fp8::reshape");
 return descriptor.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(gaudi_block_fp8,m){
 m.def("reshape(Tensor input, int[] shape) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_block_fp8::reshape","reshape",meta,
     [](const at::Stack&,size_t&bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_block_fp8,HPU,m){m.impl("reshape",reshape);}
TORCH_LIBRARY_IMPL(gaudi_block_fp8,Meta,m){
 m.impl("reshape",[](Tensor x,std::vector<int64_t> shape){auto m=meta({x,shape});return at::empty(m[0].shape,x.options());});
}
