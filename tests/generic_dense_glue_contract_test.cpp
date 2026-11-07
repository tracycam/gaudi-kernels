// Validate access axes and the Synapse layout entry point for generic kernels.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tpc_lib_api;

static Tensor tensor(TensorDataType type, std::initializer_list<uint64_t> sizes) {
  Tensor result{};
  result.geometry.dataType = type;
  result.geometry.dims = sizes.size();
  unsigned dim = 0;
  for (const auto size : sizes) {
    result.geometry.minSizes[dim] = result.geometry.maxSizes[dim] = size;
    ++dim;
  }
  return result;
}

static void check_patterns(const std::vector<Tensor>& tensors,
                           const std::vector<TensorAccessPattern>& patterns,
                           const HabanaKernelInstantiation& instance) {
  for (unsigned i = 0; i < tensors.size(); ++i) {
    const auto& access = patterns[i];
    if (access.allRequired) continue;
    bool used[MAX_TENSOR_DIM]{};
    for (unsigned dim = 0; dim < tensors[i].geometry.dims; ++dim) {
      const auto& map = access.mapping[dim];
      const unsigned axis = map.indexSpaceDim;
      if (axis >= instance.indexSpaceRank || used[axis])
        throw std::runtime_error("tensor dimensions share an index-space axis");
      used[axis] = true;
      if (map.a == 0 && (instance.indexSpaceGeometry[axis] != 1 || map.start_b != 0 ||
                         map.end_b != tensors[i].geometry.maxSizes[dim] - 1))
        throw std::runtime_error("full-width tensor dimension has an invalid access range");
    }
  }
}

static void check_kernel(pfnInstantiateTpcKernel instantiate, pfnGetSupportedDataLayout layouts,
                         const char* guid, std::vector<Tensor> inputs, std::vector<Tensor> outputs,
                         float* scalar, bool sparse_pool_reuse = false) {
  HabanaKernelParams params{};
  std::strcpy(params.guid.name, guid);
  params.inputTensorNr = inputs.size();
  params.outputTensorNr = outputs.size();
  params.inputTensors = inputs.data();
  params.outputTensors = outputs.data();
  if (scalar) {
    params.nodeParams.nodeParams = scalar;
    params.nodeParams.nodeParamsSize = sizeof(*scalar);
  }

  uint32_t count = 0;
  if (layouts(&params, nullptr, &count) != GLUE_SUCCESS || count != 1)
    throw std::runtime_error(std::string(guid) + ": layout count failed");
  std::vector<TensorDataLayout> in_layouts(inputs.size()), out_layouts(outputs.size());
  NodeDataLayouts supported{};
  supported.inputs = in_layouts.data();
  supported.outputs = out_layouts.data();
  supported.inputTensorNr = inputs.size();
  supported.outputTensorNr = outputs.size();
  if (layouts(&params, &supported, &count) != GLUE_SUCCESS || count != 1)
    throw std::runtime_error(std::string(guid) + ": layout query failed");
  for (const auto& layout : in_layouts)
    for (char axis : layout.layout)
      if (axis != 'x') throw std::runtime_error("unexpected input layout");
  for (const auto& layout : out_layouts)
    for (char axis : layout.layout)
      if (axis != 'x') throw std::runtime_error("unexpected output layout");

  std::vector<TensorAccessPattern> in_patterns(inputs.size()), out_patterns(outputs.size());
  HabanaKernelInstantiation instance{};
  instance.inputTensorAccessPattern = in_patterns.data();
  instance.outputTensorAccessPattern = out_patterns.data();
  if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER || !instance.kernel.elfSize)
    throw std::runtime_error(std::string(guid) + ": instantiation failed");
  check_patterns(inputs, in_patterns, instance);
  check_patterns(outputs, out_patterns, instance);
  if (sparse_pool_reuse) {
    if (instance.kernel.paramsNr != 0)
      throw std::runtime_error("sparse KV store has unexpected scalar parameters");
    for (unsigned i = 0; i < 2; ++i) {
      const auto& access = out_patterns.at(i);
      if (!access.sparseAccess || !access.inputReusabilityBinding ||
          access.inputsReusability != (1u << (4 + i)))
        throw std::runtime_error("sparse KV output does not require reuse of its pool input");
    }
    params.inputTensorNr = 4;
    if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_COUNT)
      throw std::runtime_error("sparse KV store accepted the wrong tensor ABI");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected provider shared library path");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    const auto instantiate = reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library, "InstantiateTpcKernel"));
    const auto layouts = reinterpret_cast<pfnGetSupportedDataLayout>(dlsym(library, "GetSupportedDataLayouts"));
    if (!instantiate || !layouts) throw std::runtime_error("required provider entry point missing");

    const auto row = tensor(DATA_BF16, {64, 4});
    const auto gamma = tensor(DATA_BF16, {64});
    float epsilon = 1e-6f;
    check_kernel(instantiate, layouts, "gk_rms_norm_bf16_v1", {row, gamma}, {row}, &epsilon);
    check_kernel(instantiate, layouts, "gk_fused_add_rms_norm_bf16_v1",
                 {row, row, gamma}, {row, row}, &epsilon);

    const auto scores = tensor(DATA_F32, {128, 4, 4});
    const auto probabilities = tensor(DATA_BF16, {128, 4, 4});
    const auto stats = tensor(DATA_F32, {4, 4});
    const auto data = tensor(DATA_F32, {64, 4, 4});
    const auto data_bf16 = tensor(DATA_BF16, {64, 4, 4});
    float scale = 0.125f;
    check_kernel(instantiate, layouts, "gk_attn_tile_softmax_bf16_v1",
                 {scores, tensor(DATA_I32, {4}), tensor(DATA_I32, {3})},
                 {probabilities, stats, stats}, &scale);
    check_kernel(instantiate, layouts, "gk_attn_online_merge_f32_v1",
                 {stats, stats, data, stats, stats, data}, {stats, stats, data}, nullptr);
    check_kernel(instantiate, layouts, "gk_attn_online_finalize_bf16_v1",
                 {stats, data}, {data_bf16}, nullptr);
    check_kernel(instantiate, layouts, "gk_attn_direct_softmax_bf16_v1",
                 {scores, tensor(DATA_I32, {4}), tensor(DATA_I32, {3})}, {probabilities}, &scale);
    const auto source_kv = tensor(DATA_BF16, {1024, 4});
    const auto head_major_pool = tensor(DATA_BF16, {128, 32768, 8});
    check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                 {source_kv, source_kv, tensor(DATA_I32, {4}), tensor(DATA_I32, {1}),
                  head_major_pool, head_major_pool}, {head_major_pool, head_major_pool}, nullptr, true);
    const auto source_kv64 = tensor(DATA_BF16, {128, 4});
    const auto head_major_pool64 = tensor(DATA_BF16, {64, 32768, 2});
    check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                 {source_kv64, source_kv64, tensor(DATA_I32, {4}), tensor(DATA_I32, {1}),
                  head_major_pool64, head_major_pool64}, {head_major_pool64, head_major_pool64}, nullptr, true);
    const auto decode_scores = tensor(DATA_F32, {1024, 1, 56});
    const auto decode_stats = tensor(DATA_F32, {64, 56});
    const auto decode_probs = tensor(DATA_BF16, {1024, 1, 56});
    check_kernel(instantiate, layouts, "gk_attn_decode_exp_f32_v1",
                 {decode_scores, tensor(DATA_I32, {1}), tensor(DATA_I32, {3})},
                 {decode_scores, decode_stats}, &scale);
    check_kernel(instantiate, layouts, "gk_attn_decode_normalize_bf16_v1",
                 {decode_scores, decode_stats}, {decode_probs}, nullptr);
    dlclose(library);
    std::cout << "generic norm and attention access axes and layouts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
