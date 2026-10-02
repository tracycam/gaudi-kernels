// SPDX-License-Identifier: Apache-2.0
// Public CustomOp registration: contributes one TPC node to the caller's graph.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using Tensor=at::Tensor;
using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta pure_meta(const at::Stack&s){
 auto x=s[0].toTensor(),w=s[1].toTensor();double eps=s[2].toDouble();
 for(auto t:{x,w})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"pure RMSNorm requires contiguous BF16 inference inputs");
 TORCH_CHECK(x.dim()==2&&w.dim()==1&&w.size(0)==x.size(1),"pure RMSNorm expects x [M,H], gamma [H]");
 TORCH_CHECK(x.size(0)>0&&x.size(0)<=INT32_MAX&&x.size(1)>0&&x.size(1)<=8192,"pure RMSNorm supports 1<=H<=8192 and positive signed-i32 M");
 TORCH_CHECK(x.device()==w.device(),"pure RMSNorm device mismatch");
 TORCH_CHECK(std::isfinite(eps)&&std::isfinite(float(eps))&&float(eps)>0,"epsilon must convert to positive finite FP32");
 return {{at::kBFloat16,x.sizes().vec()}};
}
Tensor pure(Tensor x,Tensor w,double eps){at::Stack s={x,w,eps};pure_meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_kernels::_pure_rmsnorm_bf16");return d.execute(s)[0];}
Tensor fake_pure(Tensor x,Tensor w,double eps){pure_meta({x,w,eps});return at::empty_like(x);}
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor(),r=s[1].toTensor(),w=s[2].toTensor();double eps=s[3].toDouble();
 for(auto t:{x,r,w})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"residual RMSNorm requires contiguous BF16 inference inputs");
 TORCH_CHECK(x.dim()==2&&r.sizes()==x.sizes()&&w.dim()==1&&w.size(0)==x.size(1),"residual RMSNorm expects x/residual [M,H], gamma [H]");
 TORCH_CHECK(x.size(0)>0&&x.size(0)<=INT32_MAX&&x.size(1)>0&&x.size(1)<=8192,"residual RMSNorm supports 1<=H<=8192 and positive signed-i32 M");
 TORCH_CHECK(x.device()==r.device()&&x.device()==w.device(),"residual RMSNorm device mismatch");
 TORCH_CHECK(std::isfinite(eps)&&std::isfinite(float(eps))&&float(eps)>0,"epsilon must convert to positive finite FP32");
 return {{at::kBFloat16,x.sizes().vec()},{at::kBFloat16,x.sizes().vec()}};
}
std::shared_ptr<void> params(const at::Stack&s,size_t&size){size=4;return std::make_shared<float>(s[3].toDouble());}
std::tuple<Tensor,Tensor> run(Tensor x,Tensor r,Tensor w,double eps){
 at::Stack s={x,r,w,eps};meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_kernels::_residual_rmsnorm_bf16");
 auto out=descriptor.execute(s);return {out[0],out[1]};
}
std::tuple<Tensor,Tensor> fake(Tensor x,Tensor r,Tensor w,double eps){meta({x,r,w,eps});return {at::empty_like(x),at::empty_like(x)};}
Meta reshape_meta(const at::Stack&s){
 auto x=s[0].toTensor();auto shape=s[1].toIntVector();
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&!x.requires_grad(),"norm graph reshape requires contiguous BF16 inference input");
 TORCH_CHECK(shape.size()>=1&&shape.size()<=5,"norm graph reshape rank out of range");
 int64_t count=1;for(auto n:shape){TORCH_CHECK(n>0&&n<=INT32_MAX&&count<=INT64_MAX/n,"norm graph reshape size out of range");count*=n;}
 TORCH_CHECK(count==x.numel(),"norm graph reshape must preserve elements");return {{at::kBFloat16,shape}};
}
Tensor reshape(Tensor x,c10::List<int64_t>shape){
 at::Stack s={x,shape};reshape_meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");
 auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_kernels::_norm_reshape");return d.execute(s)[0];
}
Tensor fake_reshape(Tensor x,c10::List<int64_t>shape){auto m=reshape_meta({x,shape});return at::empty(m[0].shape,x.options());}

Meta quant_meta(const at::Stack&s,bool block){
 auto result=meta(s);auto x=s[0].toTensor();int64_t m=x.size(0),h=x.size(1),g=(h+127)/128;
 result[1]={at::kFloat8_e4m3fn,block?std::vector<int64_t>{g,m,128}:std::vector<int64_t>{m,h}};
 result.push_back({at::kFloat,block?std::vector<int64_t>{g,m,1}:std::vector<int64_t>{m,1}});return result;
}
std::tuple<Tensor,Tensor,Tensor> run_quant(Tensor x,Tensor r,Tensor w,double eps,bool block){
 at::Stack s={x,r,w,eps};quant_meta(s,block);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(block?"gaudi_kernels::_residual_rmsnorm_block128_a8":"gaudi_kernels::_residual_rmsnorm_per_row_a8");
 auto out=descriptor.execute(s);return {out[0],out[1],out[2]};
}
std::tuple<Tensor,Tensor,Tensor> fake_quant(Tensor x,Tensor r,Tensor w,double eps,bool block){auto m=quant_meta({x,r,w,eps},block);return {at::empty(m[0].shape,x.options().dtype(m[0].dtype)),at::empty(m[1].shape,x.options().dtype(m[1].dtype)),at::empty(m[2].shape,x.options().dtype(m[2].dtype))};}

}
TORCH_LIBRARY_FRAGMENT(gaudi_kernels,m){
 m.def("_pure_rmsnorm_bf16(Tensor x, Tensor gamma, float epsilon) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_kernels::_pure_rmsnorm_bf16","gk_pure_rmsnorm_bf16_v1",pure_meta,[](const at::Stack&s,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<float>(s[2].toDouble());});
 m.def("_norm_reshape(Tensor input, int[] shape) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_kernels::_norm_reshape","reshape",reshape_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 m.def("_residual_rmsnorm_bf16(Tensor x, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
 m.def("_residual_rmsnorm_per_row_a8(Tensor x, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor, Tensor)");
 m.def("_residual_rmsnorm_block128_a8(Tensor x, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor, Tensor)");
 habana::custom_op::registerUserCustomOp("gaudi_kernels::_residual_rmsnorm_per_row_a8","gk_residual_rmsnorm_per_row_a8_v1",[](const at::Stack&s){return quant_meta(s,false);},params);
 habana::custom_op::registerUserCustomOp("gaudi_kernels::_residual_rmsnorm_block128_a8","gk_residual_rmsnorm_block128_a8_v1",[](const at::Stack&s){return quant_meta(s,true);},params);
 habana::custom_op::registerUserCustomOp("gaudi_kernels::_residual_rmsnorm_bf16","gk_residual_rmsnorm_bf16_v1",meta,params);
}
TORCH_LIBRARY_IMPL(gaudi_kernels,HPU,m){m.impl("_pure_rmsnorm_bf16",pure);m.impl("_norm_reshape",reshape);m.impl("_residual_rmsnorm_bf16",run);
 m.impl("_residual_rmsnorm_per_row_a8",[](Tensor x,Tensor r,Tensor w,double e){return run_quant(x,r,w,e,false);});
 m.impl("_residual_rmsnorm_block128_a8",[](Tensor x,Tensor r,Tensor w,double e){return run_quant(x,r,w,e,true);});}
TORCH_LIBRARY_IMPL(gaudi_kernels,Meta,m){m.impl("_pure_rmsnorm_bf16",fake_pure);m.impl("_norm_reshape",fake_reshape);m.impl("_residual_rmsnorm_bf16",fake);
 m.impl("_residual_rmsnorm_per_row_a8",[](Tensor x,Tensor r,Tensor w,double e){return fake_quant(x,r,w,e,false);});
 m.impl("_residual_rmsnorm_block128_a8",[](Tensor x,Tensor r,Tensor w,double e){return fake_quant(x,r,w,e,true);});}
