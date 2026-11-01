#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
using Tensor=at::Tensor;
static habana::PartialOutputMetaDataVector meta(const at::Stack&s){
  auto x=s[0].toTensor();
  TORCH_CHECK(x.scalar_type()==at::kFloat&&x.is_contiguous()&&x.numel()==64,
              "gate requires contiguous FP32[64]");
  return {{at::kFloat,{64}}};
}
TORCH_LIBRARY(gk_diagnostic_gate,m){
  m.def("delay(Tensor x) -> Tensor");
  m.def("delay_loops(Tensor x, int loops) -> Tensor");
  habana::custom_op::registerUserCustomOp("gk_diagnostic_gate::delay","gk_probe_tp8_gate_f32",meta,
    [](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<int>(50000000);});
  habana::custom_op::registerUserCustomOp("gk_diagnostic_gate::delay_loops","gk_probe_tp8_gate_f32",meta,
    [](const at::Stack&s,size_t&n)->std::shared_ptr<void>{
      auto loops=s[1].toInt();TORCH_CHECK(loops>=1&&loops<=150000000,"bounded gate loops required");
      n=4;return std::make_shared<int>(static_cast<int>(loops));});
}
TORCH_LIBRARY_IMPL(gk_diagnostic_gate,HPU,m){
  m.impl("delay",[](Tensor x){meta({x});auto descriptor=habana::custom_op::UserCustomOpDescriptor::
    getUserCustomOpDescriptor("gk_diagnostic_gate::delay");return descriptor.execute({x})[0];});
  m.impl("delay_loops",[](Tensor x,int64_t loops){meta({x});auto descriptor=habana::custom_op::UserCustomOpDescriptor::
    getUserCustomOpDescriptor("gk_diagnostic_gate::delay_loops");return descriptor.execute({x,loops})[0];});
}
