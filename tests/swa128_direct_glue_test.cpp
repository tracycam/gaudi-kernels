// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace tpc_lib_api;
extern "C" GlueCodeReturn GetKernelGuids(DeviceId, uint32_t*, GuidInfo*);
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*, HabanaKernelInstantiation*);

void expect(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void shape(Tensor& tensor, TensorDataType type, std::initializer_list<uint64_t> sizes) {
  tensor.geometry.dataType = type;
  tensor.geometry.dims = sizes.size();
  unsigned i = 0;
  for (auto size : sizes) {
    tensor.geometry.maxSizes[i] = tensor.geometry.minSizes[i] = size;
    ++i;
  }
}

int main() {
  uint32_t count = 0;
  expect(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS && count == 3,
         "unexpected GUID count");
  GuidInfo guids[3]{};
  expect(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS,
         "GUID discovery failed");
  GuidInfo direct{};
  for (const auto& guid : guids)
    if (!std::strcmp(guid.name, "gk_swa128_batch_window_direct_fp32_v3")) direct = guid;
  expect(direct.name[0] != 0, "direct GUID missing");
  for (uint32_t rows : {32u, 512u}) {
    Tensor inputs[7]{}, outputs[1]{};
    shape(inputs[0], DATA_BF16, {192, 16, rows});
    shape(inputs[1], DATA_BF16, {192, 1, 8192});
    shape(inputs[2], DATA_BF16, {128, 1, 8192});
    shape(inputs[3], DATA_I32, {2 * rows});
    shape(inputs[4], DATA_I32, {2 * rows});
    shape(inputs[5], DATA_I32, {rows});
    shape(inputs[6], DATA_BF16, {16});
    shape(outputs[0], DATA_BF16, {128, 16, rows});
    float scale = 0.07216878f;
    HabanaKernelParams params{};
    params.guid = direct;
    params.inputTensors = inputs;
    params.inputTensorNr = 7;
    params.outputTensors = outputs;
    params.outputTensorNr = 1;
    params.nodeParams.nodeParams = &scale;
    params.nodeParams.nodeParamsSize = sizeof(scale);
    TensorAccessPattern in_patterns[7]{}, out_patterns[1]{};
    HabanaKernelInstantiation instance{};
    instance.inputTensorAccessPattern = in_patterns;
    instance.outputTensorAccessPattern = out_patterns;
    expect(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER &&
               instance.kernel.elfSize > 0, "valid geometry did not expose an ELF");
    std::vector<uint8_t> elf(instance.kernel.elfSize);
    instance.kernel.kernelElf = elf.data();
    expect(InstantiateTpcKernel(&params, &instance) == GLUE_SUCCESS &&
               instance.indexSpaceGeometry[1] == rows, "valid geometry rejected");
    shape(inputs[3], DATA_I32, {2 * rows - 1});
    shape(inputs[4], DATA_I32, {2 * rows - 1});
    expect(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE,
           "non-paired page metadata accepted");
  }
}
