// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <cmath>
#include <cstring>
#include <initializer_list>

using namespace tpc_lib_api;
extern "C" unsigned char _binary_gather_o_start, _binary_gather_o_end;
extern "C" unsigned char _binary_softmax_o_start, _binary_softmax_o_end;
extern "C" unsigned char _binary_batch_gather_o_start, _binary_batch_gather_o_end;
extern "C" unsigned char _binary_batch_softmax_o_start, _binary_batch_softmax_o_end;
static const char* names[] = {"gk_full_mme_gather_v1", "gk_full_mme_softmax_v2",
                              "gk_full_mme_batch_gather_v1", "gk_full_mme_batch_softmax_v1"};

static int kind(const char* name) {
  for (int i = 0; i < 4; ++i) if (!std::strcmp(name, names[i])) return i;
  return -1;
}
static bool shape(const Tensor& t, TensorDataType type,
                  std::initializer_list<unsigned long> dims) {
  if (t.geometry.dataType != type || t.geometry.dims != dims.size()) return false;
  unsigned i = 0;
  for (auto size : dims) {
    if (t.geometry.maxSizes[i] != size || t.geometry.minSizes[i] != size) return false;
    ++i;
  }
  return true;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* info) {
  const unsigned capacity = *count;
  *count = device == DEVICE_ID_GAUDI2 ? 4 : 0;
  if (info) for (unsigned i = 0; i < capacity && i < *count; ++i) std::strcpy(info[i].name, names[i]);
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,
                                                HabanaKernelInstantiation* out) {
  const int variant = kind(in->guid.name);
  if (variant < 0) return GLUE_NODE_NOT_FOUND;
  const bool batch = variant >= 2;
  const bool gather = (variant % 2) == 0;
  if (in->inputTensorNr != (gather ? (batch ? 6u : 5u) : (batch ? 5u : 4u)) ||
      in->outputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  const unsigned page_input = gather ? 2 : 1;
  const unsigned long pages = in->inputTensors[page_input].geometry.maxSizes[0];
  if (!pages || pages > 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const unsigned long rows = batch ? in->inputTensors[page_input + 3].geometry.maxSizes[0] : 1;
  if (batch && ((rows != 4 && rows != 8) ||
                !shape(in->inputTensors[page_input + 3], DATA_I32, {rows})))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  for (unsigned i = page_input; i < page_input + 3; ++i)
    if (batch ? !shape(in->inputTensors[i], DATA_I32, {pages, rows})
              : !shape(in->inputTensors[i], DATA_I32, {pages}))
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
  if (gather) {
    const unsigned long slots = in->inputTensors[0].geometry.maxSizes[2];
    if (!slots || slots % 128 ||
        !shape(in->inputTensors[0], DATA_BF16, {192, 1, slots}) ||
        !shape(in->inputTensors[1], DATA_BF16, {128, 1, slots}) ||
        (batch ? !shape(in->outputTensors[0], DATA_BF16, {192, pages * 128, rows})
               : !shape(in->outputTensors[0], DATA_BF16, {192, pages * 128})) ||
        (batch ? !shape(in->outputTensors[1], DATA_BF16, {128, pages * 128, rows})
               : !shape(in->outputTensors[1], DATA_BF16, {128, pages * 128})) ||
        in->nodeParams.nodeParamsSize != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->kernel.paramsNr = 0;
    out->indexSpaceGeometry[0] = pages;
  } else {
    struct Params { float scale; int physical_pages; };
    if ((batch ? !shape(in->inputTensors[0], DATA_F32, {pages * 128, 16, rows})
               : !shape(in->inputTensors[0], DATA_F32, {pages * 128, 16})) ||
        (batch ? !shape(in->outputTensors[0], DATA_BF16, {pages * 128, 16, rows})
               : !shape(in->outputTensors[0], DATA_BF16, {pages * 128, 16})) ||
        (batch ? !shape(in->outputTensors[1], DATA_BF16, {pages * 128, 16, rows})
               : !shape(in->outputTensors[1], DATA_BF16, {pages * 128, 16})) ||
        in->nodeParams.nodeParamsSize != sizeof(Params) || !in->nodeParams.nodeParams)
      return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& params = *static_cast<const Params*>(in->nodeParams.nodeParams);
    if (!std::isfinite(params.scale) || params.scale <= 0 || params.physical_pages <= 0)
      return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, &params, sizeof(params));
    out->indexSpaceGeometry[0] = 16;
  }
  out->indexSpaceRank = batch ? 2 : 1;
  if (batch) out->indexSpaceGeometry[1] = rows;
  for (unsigned i = 0; i < in->inputTensorNr; ++i) out->inputTensorAccessPattern[i].allRequired = 1;
  for (unsigned i = 0; i < in->outputTensorNr; ++i) out->outputTensorAccessPattern[i].allRequired = 1;
  unsigned char* starts[] = {&_binary_gather_o_start, &_binary_softmax_o_start,
                             &_binary_batch_gather_o_start, &_binary_batch_softmax_o_start};
  unsigned char* ends[] = {&_binary_gather_o_end, &_binary_softmax_o_end,
                           &_binary_batch_gather_o_end, &_binary_batch_softmax_o_end};
  const unsigned capacity = out->kernel.elfSize;
  out->kernel.elfSize = ends[variant] - starts[variant];
  if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
  std::memcpy(out->kernel.kernelElf, starts[variant], out->kernel.elfSize);
  return GLUE_SUCCESS;
}
extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetShapeInference(DeviceId, const ShapeInferenceParams* in,
                                             ShapeInferenceOutput*) {
  return kind(in->pGuid ? in->pGuid->name : in->guid.name) < 0 ? GLUE_NODE_NOT_FOUND : GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,
                                                  NodeDataLayouts* layouts, uint32_t* count) {
  if (kind(in->guid.name) < 0) return GLUE_NODE_NOT_FOUND;
  const unsigned capacity = *count;
  *count = 1;
  if (!layouts || !capacity) return GLUE_SUCCESS;
  for (unsigned i = 0; i < in->inputTensorNr; ++i)
    std::memset(layouts[0].inputs[i].layout, 'x', sizeof(layouts[0].inputs[i].layout));
  for (unsigned i = 0; i < in->outputTensorNr; ++i)
    std::memset(layouts[0].outputs[i].layout, 'x', sizeof(layouts[0].outputs[i].layout));
  return GLUE_SUCCESS;
}
