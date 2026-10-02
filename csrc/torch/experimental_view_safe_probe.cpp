// Probe only: no production namespace or default dispatch is modified.
#include "experimental_view_safe.hpp"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <climits>
namespace {
habana::PartialOutputMetaDataVector meta(const at::Stack& stack) {
  auto x = stack[0].toTensor();
  auto shape = stack[1].toIntVector();
  TORCH_CHECK(!x.requires_grad() && shape.size() > 0 && shape.size() <= 4, "inference/static rank1..4 required");
  int64_t count = 1;
  for (auto n : shape) { TORCH_CHECK(n > 0 && n <= INT32_MAX && count <= INT64_MAX/n, "shape overflow"); count *= n; }
  TORCH_CHECK(count == x.numel(), "reshape element count mismatch");
  return {{x.scalar_type(), shape}};
}
at::Tensor reshape(at::Tensor input, std::vector<int64_t> shape, bool safe) {
  at::Stack stack{input, shape, safe};
  meta(stack);
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_view_safe::reshape");
  return (safe ? gaudi_kernels::experimental::execute_view_safe_symbols(descriptor, stack)
               : descriptor.execute(stack))[0];
}
std::vector<at::Tensor> normalize_list(std::vector<at::Tensor> inputs) {
  return gaudi_kernels::experimental::normalize_view_inputs({inputs})[0].toTensorVector();
}
}
TORCH_LIBRARY(gaudi_view_safe, m) {
  m.def("reshape(Tensor input, int[] shape, bool safe) -> Tensor");
  m.def("runtime_pin() -> str", gaudi_kernels::experimental::view_safe_runtime_pin);
  m.def("normalize_list(Tensor[] inputs) -> Tensor[]", normalize_list);
  habana::custom_op::registerUserCustomOp("gaudi_view_safe::reshape", "reshape", meta,
      [](const at::Stack&, size_t& size)->std::shared_ptr<void>{size=0;return nullptr;});
}
TORCH_LIBRARY_IMPL(gaudi_view_safe, HPU, m) { m.impl("reshape", reshape); }
TORCH_LIBRARY_IMPL(gaudi_view_safe, Meta, m) {
  m.impl("reshape", [](at::Tensor input, std::vector<int64_t> shape, bool safe) {
    auto output = meta({input, shape, safe}); return at::empty(output[0].shape, input.options());
  });
}
