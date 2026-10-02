#include <torch/extension.h>
#include <hpu_custom_op_pt2.h>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
using habana::custom_op::UserCustomOpDescriptor;using habana::custom_op::registerUserCustomOp;
static Meta gp_meta(const at::Stack&s){
 auto x=s[2].toTensor(),ids=s[4].toTensor();
 TORCH_CHECK(x.dim()==2&&x.size(1)==6144&&x.scalar_type()==at::kBFloat16);
 // Lazy TopK can retain logical Long metadata after an explicit int32 cast.
 // Device glue independently requires DATA_I32 for the actual kernel input.
 TORCH_CHECK(ids.dim()==2&&ids.size(0)==x.size(0)&&(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong),"gp metadata x=",x.sizes()," ids=",ids.sizes()," dtype=",ids.scalar_type());
 return {{at::kFloat,{1,ids.numel()*3*512}}};
}
static Meta down_meta(const at::Stack&s){
 auto x=s[2].toTensor(),ids=s[4].toTensor();
 TORCH_CHECK(x.dim()==2&&x.size(1)==256&&x.size(0)==ids.numel()&&x.scalar_type()==at::kBFloat16);
 TORCH_CHECK(ids.dim()==2&&(ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong),"down metadata ids=",ids.sizes()," dtype=",ids.scalar_type());
 return {{at::kFloat,{1,ids.numel()*6144}}};
}
static Meta gate_meta(const at::Stack&s){return {{at::kBFloat16,{s[1].toTensor().numel(),256}}};}
static std::shared_ptr<void> gate_params(const at::Stack&,size_t&size){size=4;return std::make_shared<int>(3);}
static Tensor gp(Tensor w,Tensor s,Tensor x,Tensor table,Tensor ids){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::gp");return d.execute({w,s,x,table,ids})[0];}
static Tensor down(Tensor w,Tensor s,Tensor x,Tensor table,Tensor ids){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::down");return d.execute({w,s,x,table,ids})[0];}
static Tensor gate(Tensor p,Tensor ids){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::gate");return d.execute({p,ids})[0];}
static Meta broadcast_meta(const at::Stack&s){auto n=s[1].toTensor().numel();return {{at::kBFloat16,{n*12,256}},{at::kInt,{1,n*12}}};}
static std::tuple<Tensor,Tensor> broadcast(Tensor p,Tensor ids){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::gate_broadcast");auto o=d.execute({p,ids});return {o[0],o[1]};}
static Meta sorted_meta(const at::Stack&s){auto ids=s[1].toTensor();auto n=ids.numel();return {{at::kBFloat16,{n*3,2048}},{at::kInt,{1,n*3}},{at::kInt,ids.sizes().vec()}};}
static std::shared_ptr<void> sorted_params(const at::Stack&s,size_t&size){size=12;return std::make_shared<std::array<int,3>>(std::array<int,3>{1,3,int(s[1].toTensor().size(1))});}
static std::tuple<Tensor,Tensor,Tensor> sorted_prep(Tensor x,Tensor ids,Tensor perm){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::sorted_prep");auto o=d.execute({x,ids,perm});return {o[0],o[1],o[2]};}
static Meta combine_meta(const at::Stack&s){return {{at::kFloat,{s[1].toTensor().size(0),6144}}};}
static std::shared_ptr<void> combine_params(const at::Stack&s,size_t&size){size=4;return std::make_shared<int>(s[1].toTensor().size(1));}
static Tensor combine(Tensor p,Tensor r,Tensor d,Tensor inv){auto desc=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::combine");return desc.execute({p,r,d,inv})[0];}
static Meta masked_meta(const at::Stack&s){return {{at::kFloat,{1,s[2].toTensor().size(0)*512}}};}
static Tensor masked(Tensor w,Tensor s,Tensor x,Tensor t,Tensor m){auto d=UserCustomOpDescriptor::getUserCustomOpDescriptor("unified_batch::masked_gemv");return d.execute({w,s,x,t,m})[0];}
TORCH_LIBRARY(unified_batch,m){
 m.def("gp(Tensor w, Tensor s, Tensor x, Tensor table, Tensor ids) -> Tensor");
 m.def("down(Tensor w, Tensor s, Tensor x, Tensor table, Tensor ids) -> Tensor");
 m.def("gate(Tensor partial, Tensor ids) -> Tensor");
 registerUserCustomOp("unified_batch::gp","ub_direct_gp",gp_meta,nullptr);
 registerUserCustomOp("unified_batch::down","ub_direct_down",down_meta,nullptr);
 registerUserCustomOp("unified_batch::gate","ub_compact_gate",gate_meta,gate_params);
 m.def("gate_broadcast(Tensor partial, Tensor ids) -> (Tensor, Tensor)");
 m.def("sorted_prep(Tensor x, Tensor sorted_ids, Tensor permutation) -> (Tensor, Tensor, Tensor)");
 m.def("combine(Tensor partial, Tensor routing, Tensor directions, Tensor inverse) -> Tensor");
 registerUserCustomOp("unified_batch::gate_broadcast","ub_gate_broadcast",broadcast_meta,gate_params);
 registerUserCustomOp("unified_batch::sorted_prep","ub_sorted_prep",sorted_meta,sorted_params);
 registerUserCustomOp("unified_batch::combine","ub_sorted_combine",combine_meta,combine_params);
 m.def("masked_gemv(Tensor w, Tensor s, Tensor x, Tensor table, Tensor mapping) -> Tensor");
 registerUserCustomOp("unified_batch::masked_gemv","ub_masked_gemv",masked_meta,nullptr);
}
TORCH_LIBRARY_IMPL(unified_batch,HPU,m){m.impl("gp",gp);m.impl("down",down);m.impl("gate",gate);m.impl("gate_broadcast",broadcast);m.impl("sorted_prep",sorted_prep);m.impl("combine",combine);m.impl("masked_gemv",masked);}
