// Independent, default-off public current-graph experiment. No private API.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
#include <cstdlib>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const T&t,at::ScalarType type,const char* name){TORCH_CHECK(t.scalar_type()==type&&t.dim()==2&&t.is_contiguous()&&!t.requires_grad(),"down-combine contract: ",name," actual_dtype=",t.scalar_type()," expected_dtype=",type," sizes=",t.sizes()," strides=",t.strides()," requires_grad=",t.requires_grad());TORCH_CHECK(t.numel()>0&&t.numel()<=INT_MAX,"down-combine tensor size limit");}
void same_device(const at::Stack&s){auto device=s[0].toTensor().device();for(const auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==device,"down-combine device mismatch");}
Meta meta(const at::Stack&s,bool old){
 auto w=s[0].toTensor(),scale=s[1].toTensor(),a=s[2].toTensor(),lut=s[3].toTensor(),ids=s[4].toTensor();
 check(w,at::kByte,"packed_owner_or_view");check(scale,at::kByte,"scales");check(a,at::kBFloat16,"gate_activation");check(lut,at::kBFloat16,"lookup_table");const char* i64=std::getenv("PT_ENABLE_INT64_SUPPORT");
 TORCH_CHECK(ids.scalar_type()==at::kInt||(ids.scalar_type()==at::kLong&&i64&&std::string(i64)=="0"),"route_ids require logical Int, or bridge Long with explicit physical-I32 mode");
 check(ids,ids.scalar_type(),"route_ids");same_device(s);
 int64_t width=old?256:128,rows=old?3072:6144;
 TORCH_CHECK(w.size(1)==width&&w.size(0)%rows==0&&w.size(0)/rows>=8&&w.size(0)/rows<=384&&scale.size(1)==width*2&&scale.size(0)*32==w.size(0),"historical down weight owner/view geometry");
 TORCH_CHECK(a.size(0)==8&&a.size(1)==256&&ids.size(0)==1&&ids.size(1)==8&&lut.size(0)==1&&lut.size(1)==512,"fixed M1/top8/N6144/K256 domain");
 if(!old){auto r=s[5].toTensor(),d=s[6].toTensor();check(r,at::kFloat,"routing");check(d,at::kByte,"directions");TORCH_CHECK(r.sizes()==ids.sizes()&&d.size(0)==2&&d.size(1)==256,"FP32 routing and original directions required");TORCH_CHECK(s[7].toInt()==1,"HistoricalN512 layout=1 required; native-v2 is incompatible");TORCH_CHECK(s[8].toBool(),"caller must certify original fast arithmetic and valid device route IDs");}
 return {{at::kFloat,{1,old?49152:6144}}};
}
Meta view_meta(const at::Stack&s){auto w=s[0].toTensor();bool scale=s[1].toBool();check(w,at::kByte,"packed_owner_or_view");int64_t width=scale?512:256,rows=scale?96:3072;TORCH_CHECK(w.size(1)==width&&w.size(0)%rows==0&&w.size(0)/rows>=8&&w.size(0)/rows<=384,"original historical down owner required");return {{at::kByte,{w.size(0)*2,w.size(1)/2}}};}
T invoke(const char*name,const at::Stack&s,Meta m){if(s[0].toTensor().device().type()==at::kMeta)return at::empty(m[0].shape,s[0].toTensor().options().dtype(m[0].dtype));TORCH_CHECK(s[0].toTensor().device().type()==at::kHPU,"HPU required; no CPU numerical fallback");auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return op.execute(s)[0];}
T p4(T w,T s,T a,T lut,T ids,T r,T d,int64_t layout,bool certified){at::Stack x{w,s,a,lut,ids,r,d,layout,certified};return invoke("gaudi_down_combine_isa::p4",x,meta(x,false));}
T p6(T w,T s,T a,T lut,T ids,T r,T d,int64_t layout,bool certified){at::Stack x{w,s,a,lut,ids,r,d,layout,certified};return invoke("gaudi_down_combine_isa::p6",x,meta(x,false));}
T old_body(T w,T s,T a,T lut,T ids){at::Stack x{w,s,a,lut,ids};return invoke("gaudi_down_combine_isa::old_body",x,meta(x,true));}
T weight_view(T w,bool scale){at::Stack x{w,scale};return invoke("gaudi_down_combine_isa::weight_view",x,view_meta(x));}
std::shared_ptr<void> nop(const at::Stack&,size_t&n){n=0;return nullptr;}
}
TORCH_LIBRARY(gaudi_down_combine_isa,m){
 using habana::custom_op::registerUserCustomOp;
 m.def("p4(Tensor w128, Tensor s256, Tensor gate, Tensor lut, Tensor ids, Tensor routing, Tensor directions, int layout, bool certified) -> Tensor");
 m.def("p6(Tensor w128, Tensor s256, Tensor gate, Tensor lut, Tensor ids, Tensor routing, Tensor directions, int layout, bool certified) -> Tensor");
 m.def("old_body(Tensor w256, Tensor s512, Tensor gate, Tensor lut, Tensor ids) -> Tensor");
 m.def("weight_view(Tensor owner, bool scales) -> Tensor");
 registerUserCustomOp("gaudi_down_combine_isa::p4","gk_down_combine_m1_p4_v0",[](const at::Stack&s){return meta(s,false);},nop);
 registerUserCustomOp("gaudi_down_combine_isa::p6","gk_down_combine_m1_p6_v0",[](const at::Stack&s){return meta(s,false);},nop);
 registerUserCustomOp("gaudi_down_combine_isa::old_body","gk_down_combine_old_body_v0",[](const at::Stack&s){return meta(s,true);},nop);
 registerUserCustomOp("gaudi_down_combine_isa::weight_view","reshape",view_meta,nop);
}
#define GK_DOWN_COMBINE_IMPL m.impl("p4",p4);m.impl("p6",p6);m.impl("old_body",old_body);m.impl("weight_view",weight_view);
TORCH_LIBRARY_IMPL(gaudi_down_combine_isa,HPU,m){GK_DOWN_COMBINE_IMPL}
TORCH_LIBRARY_IMPL(gaudi_down_combine_isa,Meta,m){GK_DOWN_COMBINE_IMPL}
