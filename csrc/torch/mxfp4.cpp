#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <synapse_common_types.hpp>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const Tensor&t,at::ScalarType type,int dim){
 TORCH_CHECK(t.scalar_type()==type&&t.dim()==dim&&t.is_contiguous()&&!t.requires_grad(),"MXFP4 tensor contract mismatch");
 for(auto n:t.sizes())TORCH_CHECK(n>0&&n<=INT32_MAX,"MXFP4 dimensions out of range");
}
Meta decode_meta(const at::Stack&s){
 auto w=s[0].toTensor(),e=s[1].toTensor(),lut=s[2].toTensor();auto n=s[3].toInt();
 check(w,at::kByte,3);check(e,at::kByte,3);check(lut,at::kBFloat16,2);
 TORCH_CHECK(n>0&&n<=INT32_MAX&&w.size(0)==(n+255)/256&&w.size(2)==128,"requires MXFP4 N256 layout-v1");
 TORCH_CHECK(e.size(0)==w.size(0)&&e.size(1)==(w.size(1)+31)/32&&e.size(2)==256,"MXFP4 scale layout mismatch");
 TORCH_CHECK(lut.size(0)==1&&lut.size(1)==512&&w.device()==e.device()&&w.device()==lut.device(),"MXFP4 LUT/device mismatch");
 return {{at::kBFloat16,{w.size(1),n}}};
}
Meta mm_meta(const at::Stack&s){auto x=s[0].toTensor(),w=s[1].toTensor();check(x,at::kBFloat16,2);check(w,at::kBFloat16,2);
 TORCH_CHECK(x.device()==w.device()&&x.size(1)==w.size(0),"MXFP4 MME shape/device mismatch");return {{at::kFloat,{x.size(0),w.size(1)}}};}
Meta bias_meta(const at::Stack&s){auto p=s[0].toTensor(),b=s[1].toTensor();check(p,at::kFloat,2);check(b,at::kFloat,1);
 TORCH_CHECK(p.device()==b.device()&&p.size(1)==b.size(0),"MXFP4 bias shape/device mismatch");return {{at::kFloat,p.sizes().vec()}};}
Tensor execute(const char*name,const at::Stack&s){TORCH_CHECK(s[0].toTensor().device().type()==at::kHPU,"HPU required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return d.execute(s)[0];}
Tensor decode(Tensor w,Tensor e,Tensor lut,int64_t n){decode_meta({w,e,lut,n});return execute("gaudi_kernels::_mxfp4_decode",{w,e,lut,n});}
Tensor mm(Tensor x,Tensor w){mm_meta({x,w});return execute("gaudi_kernels::_mxfp4_mm",{x,w});}
Tensor bias(Tensor p,Tensor b){bias_meta({p,b});return execute("gaudi_kernels::_mxfp4_bias",{p,b});}
Tensor fake(Tensor x,const Meta&v){return at::empty(v[0].shape,x.options().dtype(v[0].dtype));}
}
TORCH_LIBRARY_FRAGMENT(gaudi_kernels,m){
 using habana::custom_op::registerUserCustomOp;
 m.def("_mxfp4_decode(Tensor packed, Tensor scales, Tensor lut, int n) -> Tensor");
 m.def("_mxfp4_mm(Tensor x, Tensor decoded) -> Tensor");
 m.def("_mxfp4_bias(Tensor partial, Tensor bias) -> Tensor");
 registerUserCustomOp("gaudi_kernels::_mxfp4_decode","gk_mxfp4_decode_bf16_v1",decode_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<int32_t>(0);});
 registerUserCustomOp("gaudi_kernels::_mxfp4_mm","gemm",mm_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=sizeof(synGEMMParams);return std::make_shared<synGEMMParams>(false,false);});
 registerUserCustomOp("gaudi_kernels::_mxfp4_bias","gk_mxfp4_bias_f32_v1",bias_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_kernels,HPU,m){m.impl("_mxfp4_decode",decode);m.impl("_mxfp4_mm",mm);m.impl("_mxfp4_bias",bias);}
TORCH_LIBRARY_IMPL(gaudi_kernels,Meta,m){
 m.impl("_mxfp4_decode",[](Tensor w,Tensor e,Tensor lut,int64_t n){return fake(w,decode_meta({w,e,lut,n}));});
 m.impl("_mxfp4_mm",[](Tensor x,Tensor w){return fake(x,mm_meta({x,w}));});
 m.impl("_mxfp4_bias",[](Tensor p,Tensor b){return fake(p,bias_meta({p,b}));});
}
