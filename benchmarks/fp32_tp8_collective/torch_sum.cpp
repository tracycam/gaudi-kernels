#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
static Meta meta(const at::Stack&s){auto x=s[0].toTensor();TORCH_CHECK(x.scalar_type()==at::kFloat&&x.is_contiguous()&&x.dim()==2&&x.size(0)==8&&x.size(1)==6144&&!x.requires_grad(),"original FP32[8,6144] required");return {{at::kFloat,{1,6144}}};}
static Tensor sum(Tensor x){meta({x});TORCH_CHECK(x.device().type()==at::kHPU,"HPU only");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("tp8_fp32_probe::sum");return d.execute({x})[0];}
TORCH_LIBRARY(tp8_fp32_probe,m){m.def("sum(Tensor x) -> Tensor");habana::custom_op::registerUserCustomOp("tp8_fp32_probe::sum","gk_probe_tp8_sum_f32",meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<int>(8);});}
TORCH_LIBRARY_IMPL(tp8_fp32_probe,HPU,m){m.impl("sum",sum);}
TORCH_LIBRARY_IMPL(tp8_fp32_probe,Meta,m){m.impl("sum",[](Tensor x){auto m=meta({x});return at::empty(m[0].shape,x.options());});}
