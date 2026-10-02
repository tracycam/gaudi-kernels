// EXPERIMENT ONLY: this uses private, version-specific bridge headers/types.
// It is not linked into the public MoE graph reshape library.
// Source hypothesis: the vector-return LazyOp::call overload omits the input
// view update present in the single-Tensor overload. Compile/link/device gates
// must pass against the installed bridge before any caller adopts this helper.
#include "habana_kernels/lazy_kernels.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace gaudi_kernels::experimental {
std::vector<at::Tensor> execute_view_safe(
    const habana::custom_op::UserCustomOpDescriptor& descriptor,
    const std::vector<c10::IValue>& inputs) {
  std::vector<std::vector<int64_t>> shapes;
  std::vector<at::ScalarType> types;
  for (const auto& meta : descriptor.getOutputMetaFn()(inputs)) {
    shapes.push_back(meta.shape);
    types.push_back(meta.dtype);
  }
  habana_lazy::LazyOp<std::vector<at::Tensor>> op{
      descriptor.getSchemaName(), inputs, std::move(shapes)};
  op.set_scalar_types(std::move(types));
  op.viewUpdateInputs();
  return op.call();
}
} // namespace gaudi_kernels::experimental
