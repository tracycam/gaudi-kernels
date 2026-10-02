// Public graph nodes only; selection/order remain with the vendor TopK.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using T=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
struct Params{float factor;int renormalize;int apply_scale;};
Meta meta(const at::Stack&s){
 auto scores=s[0].toTensor(),ids=s[1].toTensor();double factor=s[2].toDouble();
 TORCH_CHECK(scores.dim()==2&&scores.size(0)==1&&scores.size(1)==384&&ids.dim()==2&&ids.size(0)==1&&ids.size(1)==8,"post8 supports M1 E384 K8 only");
 TORCH_CHECK(scores.scalar_type()==at::kFloat,"post8 scores must remain FP32; got ",scores.scalar_type());
 // TopkMeta declares logical Long. With native INT64 disabled, the Lazy
 // backend maps it to syn_type_int32 and can retain Long metadata even after
 // the wrapper's explicit to(int32). Do not reinterpret or change storage:
 // the independent TPC glue still rejects every physical type except DATA_I32.
 TORCH_CHECK(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong,
             "post8 ordered IDs require Int or logical Long metadata; got ",ids.scalar_type());
 TORCH_CHECK(scores.device()==ids.device()&&scores.is_contiguous()&&ids.is_contiguous()&&!scores.requires_grad()&&!ids.requires_grad(),"same-device contiguous inference tensors required");
 TORCH_CHECK(std::isfinite(factor)&&std::isfinite(float(factor)),"factor must be finite FP32");return {{at::kFloat,{1,8}},{at::kInt,{1,8}},{at::kFloat,{1,1}}};
}
std::shared_ptr<void> params(const at::Stack&s,size_t&n){n=sizeof(Params);double f=s[2].toDouble();return std::make_shared<Params>(Params{float(f),int(s[3].toBool()),int(f!=1.0)});}
std::tuple<T,T,T> run(T scores,T ids,double factor,bool renorm,bool vector){at::Stack s{scores,ids,factor,renorm};meta(s);auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(vector?"gaudi_router_post8::vector":"gaudi_router_post8::scalar");auto y=d.execute(s);return {y[0],y[1],y[2]};}
std::tuple<T,T,T> fake(T scores,T ids,double factor,bool renorm){auto m=meta({scores,ids,factor,renorm});return {at::empty(m[0].shape,scores.options()),at::empty(m[1].shape,ids.options().dtype(at::kInt)),at::empty(m[2].shape,scores.options())};}
}
TORCH_LIBRARY(gaudi_router_post8,m){
 m.def("scalar(Tensor scores, Tensor ids, float factor, bool renormalize) -> (Tensor, Tensor, Tensor)");
 m.def("vector(Tensor scores, Tensor ids, float factor, bool renormalize) -> (Tensor, Tensor, Tensor)");
 habana::custom_op::registerUserCustomOp("gaudi_router_post8::scalar","gk_router_post8_scalar_experiment",meta,params);
 habana::custom_op::registerUserCustomOp("gaudi_router_post8::vector","gk_router_post8_vector_experiment",meta,params);
}
TORCH_LIBRARY_IMPL(gaudi_router_post8,HPU,m){m.impl("scalar",[](T x,T i,double f,bool r){return run(x,i,f,r,false);});m.impl("vector",[](T x,T i,double f,bool r){return run(x,i,f,r,true);});}
TORCH_LIBRARY_IMPL(gaudi_router_post8,Meta,m){m.impl("scalar",fake);m.impl("vector",fake);}
