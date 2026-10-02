// Explicit experiment only. No collectives, global patch or device acquisition.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor(),r=s[1].toTensor(),w=s[2].toTensor();auto eps=s[3].toDouble();
 TORCH_CHECK(x.scalar_type()==at::kFloat&&r.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16,"AG FP32, residual/gamma BF16 required");
 TORCH_CHECK(r.dim()==2&&r.size(0)>0&&r.size(0)<=INT32_MAX/8&&r.size(1)>0&&r.size(1)<=8192,"residual [M,H] domain");
 TORCH_CHECK(x.dim()==2&&x.size(0)==8*r.size(0)&&x.size(1)==r.size(1)&&w.dim()==1&&w.size(0)==r.size(1),"rank-major gathered [8*M,H], residual [M,H], gamma [H]");
 for(auto t:{x,r,w})TORCH_CHECK(t.device()==r.device()&&t.is_contiguous()&&!t.requires_grad(),"same device, contiguous inference inputs required");
 TORCH_CHECK(std::isfinite(eps)&&std::isfinite(float(eps))&&float(eps)>0,"positive FP32 epsilon required");
 return {{at::kBFloat16,r.sizes().vec()},{at::kBFloat16,r.sizes().vec()}};
}
std::shared_ptr<void> params(const at::Stack&s,size_t&n){n=4;return std::make_shared<float>(s[3].toDouble());}
std::tuple<T,T> run(T x,T r,T w,double eps,bool vendor){
 at::Stack s{x,r,w,eps};meta(s);auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(vendor?"gaudi_ag_norm_experiment::vendor_boundaries":"gaudi_ag_norm_experiment::fp32_statistics");auto y=d.execute(s);return {y[0],y[1]};
}
std::tuple<T,T> fake(T x,T r,T w,double eps){meta({x,r,w,eps});return {at::empty_like(r),at::empty_like(r)};}
}
TORCH_LIBRARY(gaudi_ag_norm_experiment,m){
 m.def("vendor_boundaries(Tensor gathered, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
 m.def("fp32_statistics(Tensor gathered, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
 habana::custom_op::registerUserCustomOp("gaudi_ag_norm_experiment::vendor_boundaries","gk_ag8_residual_norm_vendor_boundaries_experiment",meta,params);
 habana::custom_op::registerUserCustomOp("gaudi_ag_norm_experiment::fp32_statistics","gk_ag8_residual_norm_fp32_statistics_experiment",meta,params);
}
TORCH_LIBRARY_IMPL(gaudi_ag_norm_experiment,HPU,m){m.impl("vendor_boundaries",[](T x,T r,T w,double e){return run(x,r,w,e,true);});m.impl("fp32_statistics",[](T x,T r,T w,double e){return run(x,r,w,e,false);});}
TORCH_LIBRARY_IMPL(gaudi_ag_norm_experiment,Meta,m){m.impl("vendor_boundaries",fake);m.impl("fp32_statistics",fake);}
