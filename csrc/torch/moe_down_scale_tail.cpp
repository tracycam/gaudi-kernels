// Offline experimental control/candidate. No production registration replacement.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cstdlib>
#include <cstring>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto w=s[0].toTensor(),sc=s[1].toTensor(),x=s[2].toTensor(),lut=s[3].toTensor(),ids=s[4].toTensor();auto device=x.device();
 TORCH_CHECK(device.type()==at::kHPU||device.type()==at::kMeta,"HPU/Meta only");
 for(const auto&v:s){auto t=v.toTensor();TORCH_CHECK(t.device()==device&&t.dim()==2&&t.is_contiguous()&&!t.requires_grad(),"contiguous same-device rank2 inference inputs required");}
 const char*flag=std::getenv("PT_ENABLE_INT64_SUPPORT");bool logicalLong=ids.scalar_type()==at::kLong&&device.type()==at::kHPU&&flag&&!std::strcmp(flag,"0");
 TORCH_CHECK((ids.scalar_type()==at::kInt||logicalLong)&&ids.size(0)>=1&&ids.size(0)<=8&&ids.size(1)==8,"experimental down supports T1..8 top8 only");
 auto slots=ids.numel();auto e=w.size(0)/3072;
 TORCH_CHECK(w.scalar_type()==at::kByte&&w.size(1)==256&&e>=1&&e<=384&&w.size(0)==e*3072,"original HistoricalN512 down byte geometry required");
 TORCH_CHECK(sc.scalar_type()==at::kByte&&sc.size(0)==e*96&&sc.size(1)==512,"original E8M0 geometry required");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.size(0)==slots&&x.size(1)==256&&lut.scalar_type()==at::kBFloat16&&lut.size(0)==1&&lut.size(1)==512,"BF16 gate/table geometry required");
 return{{at::kFloat,{1,slots*6144}}};
}
T call(T w,T sc,T x,T lut,T ids,bool candidate){TORCH_CHECK(ids.scalar_type()==at::kInt,"public IDs must be int32");at::Stack s{w,sc,x,lut,ids};auto m=meta(s);if(x.device().type()==at::kMeta)return at::empty(m[0].shape,x.options().dtype(at::kFloat));auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(candidate?"gaudi_down_scale_tail::candidate":"gaudi_down_scale_tail::control");return op.execute(s)[0];}
}
TORCH_LIBRARY(gaudi_down_scale_tail,m){
 m.def("control(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");m.def("candidate(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_down_scale_tail::control","gk_moe_down_289_control_experiment",meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 habana::custom_op::registerUserCustomOp("gaudi_down_scale_tail::candidate","gk_moe_down_242_scale_tail_experiment",meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
}
#define GK_DOWN_IMPL m.impl("control",[](T w,T s,T x,T t,T ids){return call(w,s,x,t,ids,false);});m.impl("candidate",[](T w,T s,T x,T t,T ids){return call(w,s,x,t,ids,true);});
TORCH_LIBRARY_IMPL(gaudi_down_scale_tail,HPU,m){GK_DOWN_IMPL}
TORCH_LIBRARY_IMPL(gaudi_down_scale_tail,Meta,m){GK_DOWN_IMPL}
