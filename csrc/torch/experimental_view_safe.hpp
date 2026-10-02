#pragma once
// Private-ABI experiment only. No object-layout declarations or interposition.
#include <hpu_custom_op_pt2.h>
namespace gaudi_kernels::experimental {
std::vector<at::Tensor> execute_view_safe_symbols(
    habana::custom_op::UserCustomOpDescriptor descriptor,
    const std::vector<c10::IValue>& inputs);
std::vector<c10::IValue> normalize_view_inputs(
    const std::vector<c10::IValue>& inputs);
std::string view_safe_runtime_pin();
}
