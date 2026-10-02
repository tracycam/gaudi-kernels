// Public framework-current-graph bridge. No acquire, mark_step or private launch.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
static Meta meta(const at::Stack&s,bool debug=false){auto p=s[0].toTensor(),a=s[1].toTensor(),w=s[2].toTensor(),b=s[3].toTensor();for(auto&t:{p,a,w,b})TORCH_CHECK(t.scalar_type()==at::kFloat&&t.is_contiguous()&&!t.requires_grad()&&t.device()==p.device(),"original contiguous FP32 operands on one device required");
 TORCH_CHECK(p.dim()==3&&a.dim()==3&&w.dim()==2&&b.dim()==1&&p.size(0)>0&&p.size(1)>0&&p.size(2)>0,"invalid reduction ranks");TORCH_CHECK(a.size(0)==p.size(0)&&a.size(1)==p.size(1)&&a.size(2)==1&&w.size(0)==(p.size(2)+127)/128&&w.size(1)==p.size(0)&&b.size(0)==p.size(2),"invalid reduction dimensions");for(auto&t:{p,a,w,b})for(auto size:t.sizes())TORCH_CHECK(size>0&&size<=INT32_MAX-128,"unsupported empty/oversize reduction shape");return {{debug?at::kFloat:at::kBFloat16,{p.size(1),p.size(2)}}};}
static Tensor execute(const char*name,Tensor p,Tensor a,Tensor w,Tensor b,bool debug){meta({p,a,w,b},debug);TORCH_CHECK(p.device().type()==at::kHPU,"HPU required");auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return descriptor.execute({p,a,w,b})[0];}
// Namespace and GUIDs are separate from the qualified production Neumaier.
#define EACH(X) \
 X(baseline,"gk_neumaier_isa_baseline_v1",false) \
 X(lookahead,"gk_neumaier_isa_lookahead_v1",false) \
 X(handschedule,"gk_neumaier_isa_handschedule_v1",false) \
 X(baseline_debug,"gk_neumaier_isa_baseline_debug_v1",true) \
 X(lookahead_debug,"gk_neumaier_isa_lookahead_debug_v1",true) \
 X(handschedule_debug,"gk_neumaier_isa_handschedule_debug_v1",true)
TORCH_LIBRARY(gk_reduce_isa,m){
#define REGISTER(N,G,D) m.def(#N "(Tensor partial, Tensor a, Tensor w, Tensor bias) -> Tensor");habana::custom_op::registerUserCustomOp("gk_reduce_isa::" #N,G,[](const at::Stack&s){return meta(s,D);},[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 EACH(REGISTER)
#undef REGISTER
}
TORCH_LIBRARY_IMPL(gk_reduce_isa,HPU,m){
#define REGISTER(N,G,D) m.impl(#N,[](Tensor p,Tensor a,Tensor w,Tensor b){return execute("gk_reduce_isa::" #N,p,a,w,b,D);});
 EACH(REGISTER)
#undef REGISTER
}
TORCH_LIBRARY_IMPL(gk_reduce_isa,Meta,m){
#define REGISTER(N,G,D) m.impl(#N,[](Tensor p,Tensor a,Tensor w,Tensor b){auto v=meta({p,a,w,b},D);return at::empty(v[0].shape,p.options().dtype(v[0].dtype));});
 EACH(REGISTER)
#undef REGISTER
}
