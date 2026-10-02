#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
static Meta meta(const at::Stack&s){auto p=s[0].toTensor(),a=s[1].toTensor(),w=s[2].toTensor(),b=s[3].toTensor();for(auto&t:{p,a,w,b})TORCH_CHECK(t.scalar_type()==at::kFloat&&t.is_contiguous()&&!t.requires_grad()&&t.device()==p.device(),"original contiguous FP32 operands on one device required");
 TORCH_CHECK(p.dim()==3&&a.dim()==3&&w.dim()==2&&b.dim()==1&&p.size(0)>0&&p.size(1)>0&&p.size(2)>0,"invalid reduction ranks");TORCH_CHECK(a.size(0)==p.size(0)&&a.size(1)==p.size(1)&&a.size(2)==1&&w.size(0)==(p.size(2)+127)/128&&w.size(1)==p.size(0)&&b.size(0)==p.size(2),"invalid reduction dimensions");return {{at::kBFloat16,{p.size(1),p.size(2)}}};}
static Tensor execute(const char*name,Tensor p,Tensor a,Tensor w,Tensor b){meta({p,a,w,b});TORCH_CHECK(p.device().type()==at::kHPU,"HPU required");auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return descriptor.execute({p,a,w,b})[0];}
#define EACH(X) X(original,"gk_block128_reduce_original_diagnostic_v1") X(chain4,"gk_block128_reduce_chain4_row1_v1") X(chain8,"gk_block128_reduce_chain8_row1_v1") X(neumaier,"gk_block128_reduce_neumaier_row1_v1")
TORCH_LIBRARY(gk_reduce_experiment,m){
#define REGISTER(N,G) m.def(#N "(Tensor partial, Tensor a, Tensor w, Tensor bias) -> Tensor");habana::custom_op::registerUserCustomOp("gk_reduce_experiment::" #N,G,meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 EACH(REGISTER)
#undef REGISTER
}
TORCH_LIBRARY_IMPL(gk_reduce_experiment,HPU,m){
#define REGISTER(N,G) m.impl(#N,[](Tensor p,Tensor a,Tensor w,Tensor b){return execute("gk_reduce_experiment::" #N,p,a,w,b);});
 EACH(REGISTER)
#undef REGISTER
}
TORCH_LIBRARY_IMPL(gk_reduce_experiment,Meta,m){
#define REGISTER(N,G) m.impl(#N,[](Tensor p,Tensor a,Tensor w,Tensor b){auto v=meta({p,a,w,b});return at::empty(v[0].shape,p.options().dtype(at::kBFloat16));});
 EACH(REGISTER)
#undef REGISTER
}
