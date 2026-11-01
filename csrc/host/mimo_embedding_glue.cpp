// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cstring>

using namespace tpc_lib_api;

extern "C" unsigned char _binary_mimo_embedding_o_start;
extern "C" unsigned char _binary_mimo_embedding_o_end;

namespace {
constexpr const char* kName = "gk_mimo_embedding_bf16_v1";
bool named(const char* name) { return name && std::strcmp(name, kName) == 0; }

bool fixed_shape(const TensorGeometry& geometry, uint32_t dims,
                 uint64_t first, uint64_t second) {
  if (geometry.dims != dims || geometry.maxSizes[0] != first ||
      (dims == 2 && geometry.maxSizes[1] != second)) return false;
  for (uint32_t dim = 0; dim < dims; ++dim) {
    if (geometry.minSizes[dim] != geometry.maxSizes[dim]) return false;
  }
  return true;
}
}  // namespace

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count,
                                           GuidInfo* guids) {
  const uint32_t capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
  if (guids && capacity && *count) std::strcpy(guids[0].name, kName);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output) {
  if (!named(input->guid.name)) return GLUE_NODE_NOT_FOUND;
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize != 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& table = input->inputTensors[0].geometry;
  const auto& token = input->inputTensors[1].geometry;
  const auto& embedding = input->outputTensors[0].geometry;
  const auto& status = input->outputTensors[1].geometry;
  if (table.dataType != DATA_BF16 || token.dataType != DATA_I32 ||
      embedding.dataType != DATA_BF16 || status.dataType != DATA_I32)
    return GLUE_INCOMPATIBLE_DATA_TYPE;
  const auto rows = token.maxSizes[0];
  if ((rows != 1 && rows != 32) || !fixed_shape(table, 2, 6144, 152576) ||
      !fixed_shape(token, 1, rows, 0))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (!fixed_shape(embedding, 2, 6144, rows) || !fixed_shape(status, 1, rows, 0))
    return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
  output->indexSpaceRank = 1;
  output->indexSpaceGeometry[0] = 48 * rows;
  output->inputTensorAccessPattern[0].allRequired = 1;
  output->inputTensorAccessPattern[1].allRequired = 1;
  output->outputTensorAccessPattern[0].allRequired = 1;
  output->outputTensorAccessPattern[1].allRequired = 1;
  output->kernel.paramsNr = 0;
  const unsigned size = static_cast<unsigned>(&_binary_mimo_embedding_o_end -
                                               &_binary_mimo_embedding_o_start);
  const unsigned capacity = output->kernel.elfSize;
  output->kernel.elfSize = size;
  if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(output->kernel.kernelElf, &_binary_mimo_embedding_o_start, size);
  return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* input,
                                              ShapeInferenceOutput*) {
  return named(input->pGuid ? input->pGuid->name : input->guid.name)
             ? GLUE_SUCCESS : GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* input,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (!named(input->guid.name)) return GLUE_NODE_NOT_FOUND;
  const auto capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned index = 0; index < 2; ++index) {
    std::memset(layouts[0].inputs[index].layout, 'x', sizeof(layouts[0].inputs[index].layout));
    std::memset(layouts[0].outputs[index].layout, 'x', sizeof(layouts[0].outputs[index].layout));
  }
  return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }
