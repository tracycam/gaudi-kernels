// SPDX-License-Identifier: Apache-2.0
// Functional small Q/K/V outputs only. Public API has no KV alias binding.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor(),c=s[1].toTensor(),t=s[2].toTensor();double scale=s[3].toDouble();
 for(auto v:{x,c,t})TORCH_CHECK(v.scalar_type()==at::kBFloat16&&v.is_contiguous()&&!v.requires_grad(),"QKV postprocess requires contiguous BF16 inference inputs");
 TORCH_CHECK(x.dim()==2&&x.size(1)==3392&&x.size(0)>0&&x.size(0)<=INT32_MAX,"expected static QKV [M,3392]");
 TORCH_CHECK(c.dim()==3&&c.size(0)==x.size(0)&&c.size(1)==1&&c.size(2)==64&&t.sizes()==c.sizes(),"expected selected cos/sin [M,1,64]");
 TORCH_CHECK(x.device()==c.device()&&x.device()==t.device()&&std::isfinite(scale)&&std::isfinite(float(scale)),"device mismatch or nonfinite value_scale");
 return {{at::kBFloat16,{x.size(0),16,192}},{at::kBFloat16,{x.size(0),1,192}},{at::kBFloat16,{x.size(0),1,128}}};
}
std::shared_ptr<void> params(const at::Stack&s,size_t&bytes){bytes=4;return std::make_shared<float>(s[3].toDouble());}
std::tuple<Tensor,Tensor,Tensor> run(const char*name,Tensor x,Tensor c,Tensor s,double scale){at::Stack stack={x,c,s,scale};meta(stack);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);auto out=d.execute(stack);return {out[0],out[1],out[2]};}
std::tuple<Tensor,Tensor,Tensor> fake(Tensor x,Tensor c,Tensor s,double scale){auto m=meta({x,c,s,scale});return {at::empty(m[0].shape,x.options()),at::empty(m[1].shape,x.options()),at::empty(m[2].shape,x.options())};}
Meta cache_meta(const at::Stack&s){
 auto x=s[0].toTensor(),c=s[1].toTensor(),p=s[2].toTensor();double scale=s[3].toDouble();
 for(auto t:{x,c})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"QKV/cache must be contiguous inference BF16");
 TORCH_CHECK(x.dim()==2&&x.size(1)==3392&&x.size(0)>0&&x.size(0)<=INT32_MAX/3392,"QKV cache interface requires bounded [M,3392]");
 TORCH_CHECK(c.dim()==2&&c.size(1)==64&&c.size(0)>0&&c.size(0)<=INT32_MAX/64,"actual layer cosine/sine cache must be [P,64]");
 TORCH_CHECK((p.scalar_type()==at::kInt||p.scalar_type()==at::kLong)&&p.is_contiguous()&&p.numel()==x.size(0)&&
  (p.dim()==1||(p.dim()==2&&(p.size(0)==1||p.size(1)==1))),"positions must be [M], [1,M] or [M,1], native I32");
 TORCH_CHECK(x.device()==c.device()&&x.device()==p.device()&&std::isfinite(scale)&&std::isfinite(float(scale)),"device mismatch or nonfinite V scale");
 return {{at::kBFloat16,{x.size(0),3072}},{at::kBFloat16,{x.size(0),1,192}},{at::kBFloat16,{x.size(0),1,128}}};
}
std::tuple<Tensor,Tensor,Tensor> cache_run(Tensor x,Tensor c,Tensor p,double scale){at::Stack s={x,c,p,scale};cache_meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU input required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_kernels::_qkv_post_cache_bf16_v2");auto out=d.execute(s);return {out[0],out[1],out[2]};}
std::tuple<Tensor,Tensor,Tensor> cache_fake(Tensor x,Tensor c,Tensor p,double scale){auto m=cache_meta({x,c,p,scale});return {at::empty(m[0].shape,x.options()),at::empty(m[1].shape,x.options()),at::empty(m[2].shape,x.options())};}
}
#define VARIANTS(X) X(bf16_bf16) X(bf16_f32) X(f32_bf16) X(f32_f32)
TORCH_LIBRARY_FRAGMENT(gaudi_kernels,m){
 m.def("_qkv_post_cache_bf16_v2(Tensor qkv, Tensor cosine_sine_cache, Tensor positions, float value_scale) -> (Tensor, Tensor, Tensor)");habana::custom_op::registerUserCustomOp("gaudi_kernels::_qkv_post_cache_bf16_v2","gk_qkv_post_cache_bf16_v2",cache_meta,params);
#define REGISTER(n) m.def("_qkv_post_" #n "(Tensor qkv, Tensor cosine, Tensor sine, float value_scale) -> (Tensor, Tensor, Tensor)");habana::custom_op::registerUserCustomOp("gaudi_kernels::_qkv_post_" #n,"gk_qkv_post_" #n "_v1",meta,params);
 VARIANTS(REGISTER)
}
TORCH_LIBRARY_IMPL(gaudi_kernels,HPU,m){
 m.impl("_qkv_post_cache_bf16_v2",cache_run);
#define IMPLEMENT(n) m.impl("_qkv_post_" #n,[](Tensor x,Tensor c,Tensor s,double scale){return run("gaudi_kernels::_qkv_post_" #n,x,c,s,scale);});
 VARIANTS(IMPLEMENT)
}
TORCH_LIBRARY_IMPL(gaudi_kernels,Meta,m){
 m.impl("_qkv_post_cache_bf16_v2",cache_fake);
#define FAKE(n) m.impl("_qkv_post_" #n,fake);
 VARIANTS(FAKE)
}
