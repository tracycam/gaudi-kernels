// Explicit experimental namespace; never replaces precision_fix::combine.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto p=s[0].toTensor(),r=s[1].toTensor(),d=s[2].toTensor();auto h=s[3].toInt();auto device=p.device();
 TORCH_CHECK(device.type()==at::kHPU||device.type()==at::kMeta,"HPU/Meta only");
 for(auto&t:{p,r,d})TORCH_CHECK(t.device()==device&&t.dim()==2&&t.is_contiguous()&&!t.requires_grad(),"same-device contiguous rank2 inference tensors required");
 TORCH_CHECK(h>0&&h<=6144&&h%128==0&&r.size(0)>=1&&r.size(0)<=8&&r.size(1)==8,"experimental T1..8/top8/N128-multiple geometry required");
 TORCH_CHECK(p.scalar_type()==at::kFloat&&r.scalar_type()==at::kFloat&&d.scalar_type()==at::kByte,"FP32 partial/routing and U8 directions required");
 TORCH_CHECK(p.size(0)==1&&p.size(1)==r.size(0)*8*h&&d.size(0)==2&&d.size(1)==256,"original even/odd partial and direction layout required");
 return{{at::kFloat,{r.size(0),h}}};
}
const char*names[]={"gaudi_combine_preload::control","gaudi_combine_preload::rolled","gaudi_combine_preload::unroll8"};
T call(T p,T r,T d,int64_t h,int which){at::Stack s{p,r,d,h};auto m=meta(s);if(p.device().type()==at::kMeta)return at::empty(m[0].shape,p.options());auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[which]);return descriptor.execute(s)[0];}
}
TORCH_LIBRARY(gaudi_combine_preload,m){
 const char*guids[]={"gk_moe_combine_scalar_control_experiment","gk_moe_combine_vector_routes_experiment","gk_moe_combine_vector_routes_unroll8_experiment"};
 m.def("control(Tensor partial, Tensor routing, Tensor directions, int h) -> Tensor");
 m.def("rolled(Tensor partial, Tensor routing, Tensor directions, int h) -> Tensor");
 m.def("unroll8(Tensor partial, Tensor routing, Tensor directions, int h) -> Tensor");
 for(int i=0;i<3;++i)habana::custom_op::registerUserCustomOp(names[i],guids[i],meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
}
#define GK_COMBINE_IMPL m.impl("control",[](T p,T r,T d,int64_t h){return call(p,r,d,h,0);});m.impl("rolled",[](T p,T r,T d,int64_t h){return call(p,r,d,h,1);});m.impl("unroll8",[](T p,T r,T d,int64_t h){return call(p,r,d,h,2);});
TORCH_LIBRARY_IMPL(gaudi_combine_preload,HPU,m){GK_COMBINE_IMPL}
TORCH_LIBRARY_IMPL(gaudi_combine_preload,Meta,m){GK_COMBINE_IMPL}
