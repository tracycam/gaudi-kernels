// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace tpc_lib_api;
extern "C" GlueCodeReturn GetKernelGuids(DeviceId, uint32_t*, GuidInfo*);
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*, HabanaKernelInstantiation*);

void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void shape(Tensor& tensor, TensorDataType type, uint64_t first, uint64_t second = 0) {
  tensor.geometry.dataType = type;
  tensor.geometry.dims = second ? 2 : 1;
  tensor.geometry.maxSizes[0] = tensor.geometry.minSizes[0] = first;
  if (second) tensor.geometry.maxSizes[1] = tensor.geometry.minSizes[1] = second;
}

int main() {
  uint32_t count = 0;
  expect(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS && count == 1,
         "unexpected GUID count");
  GuidInfo guid{};
  expect(GetKernelGuids(DEVICE_ID_GAUDI2, &count, &guid) == GLUE_SUCCESS &&
         std::strcmp(guid.name, "gk_mimo_embedding_bf16_v1") == 0, "unexpected GUID");
  Tensor inputs[2]{}, outputs[2]{};
  shape(inputs[0], DATA_BF16, 6144, 152576);
  shape(inputs[1], DATA_I32, 1);
  shape(outputs[0], DATA_BF16, 6144, 1);
  shape(outputs[1], DATA_I32, 1);
  HabanaKernelParams params{};
  params.guid = guid;
  params.inputTensors = inputs;
  params.inputTensorNr = 2;
  params.outputTensors = outputs;
  params.outputTensorNr = 2;
  TensorAccessPattern in_patterns[2]{}, out_patterns[2]{};
  HabanaKernelInstantiation instance{};
  instance.inputTensorAccessPattern = in_patterns;
  instance.outputTensorAccessPattern = out_patterns;
  expect(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER &&
         instance.kernel.elfSize > 0, "missing ELF size");
  std::vector<uint8_t> elf(instance.kernel.elfSize);
  instance.kernel.kernelElf = elf.data();
  expect(InstantiateTpcKernel(&params, &instance) == GLUE_SUCCESS, "valid geometry rejected");
  expect(instance.indexSpaceRank == 1 && instance.indexSpaceGeometry[0] == 48 &&
         out_patterns[0].mapping[0].a == 128, "incorrect index-space mapping");
  inputs[1].geometry.dataType = DATA_BF16;
  expect(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_DATA_TYPE,
         "invalid token dtype accepted");
  inputs[1].geometry.dataType = DATA_I32;
  outputs[1].geometry.maxSizes[0] = 2;
  expect(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_OUTPUT_SIZE,
         "invalid status geometry accepted");
}
