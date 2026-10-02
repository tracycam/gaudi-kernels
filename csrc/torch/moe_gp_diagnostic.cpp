// Experimental constant-one diagnostic and separate general broadcast GP.
// Weight bytes, paired lookup, K32 FP32 MAC order and scales stay unchanged.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using T=at::Tensor;
namespace {
habana::PartialOutputMetaDataVector meta(const at::Stack& s) {
  auto x=s[2].toTensor(), ids=s[4].toTensor();
  TORCH_CHECK(x.dim()==2 && x.size(1)==6144 && x.scalar_type()==at::kBFloat16,
              "experimental GP requires BF16[T,6144]");
  TORCH_CHECK(ids.dim()==2 && ids.size(0)==x.size(0) && ids.size(1)>=1 && ids.size(1)<=384,
              "diagnostic route shape");
  return {{at::kFloat,{1,ids.numel()*1536}}};
}
T run_guid(T w,T s,T x,T table,T ids,const char*schema) {
  at::Stack stack{w,s,x,table,ids};meta(stack);
  auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      schema);
  return descriptor.execute(stack)[0];
}
T run(T w,T s,T x,T table,T ids){return run_guid(w,s,x,table,ids,"gaudi_gp_diagnostic::immediate_one");}
T broadcast(T w,T s,T x,T table,T ids){return run_guid(w,s,x,table,ids,"gaudi_gp_diagnostic::broadcast");}
}
TORCH_LIBRARY(gaudi_gp_diagnostic,m) {
  m.def("immediate_one(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
  habana::custom_op::registerUserCustomOp("gaudi_gp_diagnostic::immediate_one",
      "diag_direct_gp_imm1",meta,[](const at::Stack&,size_t& bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});
  m.def("broadcast(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor");
  habana::custom_op::registerUserCustomOp("gaudi_gp_diagnostic::broadcast",
      "diag_direct_gp_broadcast",meta,[](const at::Stack&,size_t& bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_gp_diagnostic,HPU,m){m.impl("immediate_one",run);m.impl("broadcast",broadcast);}
TORCH_LIBRARY_IMPL(gaudi_gp_diagnostic,Meta,m){
  m.impl("immediate_one",[](T w,T s,T x,T table,T ids){return at::empty(meta({w,s,x,table,ids})[0].shape,x.options().dtype(at::kFloat));});
  m.impl("broadcast",[](T w,T s,T x,T table,T ids){return at::empty(meta({w,s,x,table,ids})[0].shape,x.options().dtype(at::kFloat));});
}
