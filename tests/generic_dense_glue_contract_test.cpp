// Validate access axes and the Synapse layout entry point for generic kernels.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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
                         void* scalar, bool sparse_pool_reuse = false) {
  HabanaKernelParams params{};
  std::strcpy(params.guid.name, guid);
  params.inputTensorNr = inputs.size();
  params.outputTensorNr = outputs.size();
  params.inputTensors = inputs.data();
  params.outputTensors = outputs.data();
  if (scalar) {
    params.nodeParams.nodeParams = scalar;
    params.nodeParams.nodeParamsSize = 4;
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
    const auto& access = out_patterns.at(0);
    if (out_patterns.size() != 1 || !access.sparseAccess || !access.inputReusabilityBinding ||
        access.inputsReusability != (1u << 4))
      throw std::runtime_error("sparse KV output does not require reuse of its pool input");
    params.inputTensorNr = 4;
    if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_COUNT)
      throw std::runtime_error("sparse KV store accepted the wrong tensor ABI");
    params.inputTensorNr = 5;
    params.outputTensorNr = 2;
    if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_OUTPUT_COUNT)
      throw std::runtime_error("sparse KV store accepted the wrong output ABI");
    params.outputTensorNr = 1;
    inputs[3] = tensor(DATA_I32, {1});
    if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_SIZE)
      throw std::runtime_error("sparse KV store accepted incomplete metadata");
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
    const auto stats = tensor(DATA_F32, {64, 4});
    const auto data = tensor(DATA_F32, {64, 4, 4});
    const auto data_bf16 = tensor(DATA_BF16, {64, 4, 4});
    float scale = 0.125f;
    check_kernel(instantiate, layouts, "gk_attn_tile_softmax_bf16_v1",
                 {scores, tensor(DATA_I32, {4}), tensor(DATA_I32, {3})},
                 {probabilities, stats, stats}, &scale);
    check_kernel(instantiate, layouts, "gk_attn_tile_softmax_bf16_v1",
                 {tensor(DATA_F32, {512, 256, 56}), tensor(DATA_I32, {256}), tensor(DATA_I32, {3})},
                 {tensor(DATA_BF16, {512, 256, 56}), tensor(DATA_F32, {256, 56}), tensor(DATA_F32, {256, 56})},
                 &scale);
    check_kernel(instantiate, layouts, "gk_attn_online_merge_f32_v1",
                 {stats, stats, data, stats, stats, data}, {stats, stats, data}, nullptr);
    check_kernel(instantiate, layouts, "gk_attn_online_finalize_bf16_v1",
                 {stats, data}, {data_bf16}, nullptr);
    check_kernel(instantiate, layouts, "gk_attn_direct_softmax_bf16_v1",
                 {scores, tensor(DATA_I32, {4}), tensor(DATA_I32, {3})}, {probabilities}, &scale);

    const auto scores4d = tensor(DATA_F32, {512, 512, 8, 8});
    const auto probs4d = tensor(DATA_BF16, {512, 512, 8, 8});
    const auto stats4d = tensor(DATA_F32, {512, 8, 8});
    const auto data4d = tensor(DATA_F32, {128, 512, 8, 8});
    const auto data4d_bf16 = tensor(DATA_BF16, {128, 512, 8, 8});
    check_kernel(instantiate, layouts, "gk_attn_tile_softmax_bf16_v1",
                 {scores4d, tensor(DATA_I32, {512}), tensor(DATA_I32, {3})},
                 {probs4d, stats4d, stats4d}, &scale);
    check_kernel(instantiate, layouts, "gk_attn_direct_softmax_bf16_v1",
                 {scores4d, tensor(DATA_I32, {512}), tensor(DATA_I32, {3})},
                 {probs4d}, &scale);
    int32_t prev_unscaled = 1;
    check_kernel(instantiate, layouts, "gk_attn_online_merge_f32_v1",
                 {stats4d, stats4d, data4d, stats4d, stats4d, data4d},
                 {stats4d, stats4d, data4d}, nullptr);
    check_kernel(instantiate, layouts, "gk_attn_online_merge_f32_v1",
                 {stats4d, stats4d, data4d, stats4d, stats4d, data4d},
                 {stats4d, stats4d, data4d}, &prev_unscaled);
    check_kernel(instantiate, layouts, "gk_attn_online_finalize_bf16_v1",
                 {stats4d, data4d}, {data4d_bf16}, nullptr);
    const auto source_kv = tensor(DATA_BF16, {1024, 4});
    const auto head_major_pool = tensor(DATA_BF16, {128, 2 * 131072 * 8});
    check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                 {source_kv, source_kv, tensor(DATA_I32, {4}), tensor(DATA_I32, {3}), head_major_pool},
                 {head_major_pool}, nullptr, true);
    const auto source_kv64 = tensor(DATA_BF16, {128, 4});
    const auto head_major_pool64 = tensor(DATA_BF16, {64, 2 * 131072 * 2});
    check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                 {source_kv64, source_kv64, tensor(DATA_I32, {4}), tensor(DATA_I32, {3}), head_major_pool64},
                 {head_major_pool64}, nullptr, true);
    for (const auto [keys, heads] : {std::pair{1024u, 56u}, std::pair{8192u, 4u},
                                      std::pair{8192u, 32u}, std::pair{8192u, 256u}}) {
      const auto decode_scores = tensor(DATA_F32, {keys, 1, heads});
      const auto decode_stats = tensor(DATA_F32, {64, heads});
      const auto decode_probs = tensor(DATA_BF16, {keys, 1, heads});
      check_kernel(instantiate, layouts, "gk_attn_decode_exp_f32_v1",
                   {decode_scores, tensor(DATA_I32, {1}), tensor(DATA_I32, {3})},
                   {decode_scores, decode_stats}, &scale);
      check_kernel(instantiate, layouts, "gk_attn_decode_normalize_bf16_v1",
                   {decode_scores, decode_stats}, {decode_probs}, nullptr);
    }
    {
      std::vector<Tensor> inputs{tensor(DATA_F32, {8320, 1, 4}), tensor(DATA_I32, {1}),
                                 tensor(DATA_I32, {3})};
      std::vector<Tensor> outputs{tensor(DATA_F32, {8320, 1, 4}), tensor(DATA_F32, {64, 4})};
      HabanaKernelParams params{};
      std::strcpy(params.guid.name, "gk_attn_decode_exp_f32_v1");
      params.inputTensorNr = inputs.size();
      params.outputTensorNr = outputs.size();
      params.inputTensors = inputs.data();
      params.outputTensors = outputs.data();
      params.nodeParams.nodeParams = &scale;
      params.nodeParams.nodeParamsSize = sizeof(scale);
      HabanaKernelInstantiation instance{};
      if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_SIZE)
        throw std::runtime_error("decode accepted a KV tile beyond 8192");
    }
    auto dynamic_map = tensor(DATA_I32, {131072});
    dynamic_map.geometry.minSizes[0] = 1;
    check_kernel(instantiate, layouts, "gk_attn_tile_indices_i32_v1",
                 {dynamic_map, tensor(DATA_I32, {8}), tensor(DATA_I32, {4, 16}),
                  tensor(DATA_I32, {1})}, {tensor(DATA_I32, {64, 16})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_act_row_scale_f32_v1",
                 {tensor(DATA_BF16, {256, 128})}, {tensor(DATA_F32, {1, 128})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_weight_blk_scale_f32_v1",
                 {tensor(DATA_F32, {2, 4})}, {tensor(DATA_F32, {1, 4})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_quant_act_fp16_1k_v1",
                 {tensor(DATA_BF16, {256, 128}), tensor(DATA_F32, {1, 128})},
                 {tensor(DATA_F16, {256, 128})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_dequant_weight_fp16_1k_v1",
                 {tensor(DATA_F8_143, {128, 512, 2}), tensor(DATA_F32, {2, 4}), tensor(DATA_F32, {1, 4})},
                 {tensor(DATA_F16, {256, 512})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_epilogue_bf16_v1",
                 {tensor(DATA_F32, {512, 128}), tensor(DATA_F32, {1, 128}),
                  tensor(DATA_F32, {1, 4})},
                 {tensor(DATA_BF16, {512, 128})}, nullptr);
    check_kernel(instantiate, layouts, "gk_fp8_epilogue_bias_bf16_v1",
                 {tensor(DATA_F32, {512, 128}), tensor(DATA_F32, {1, 128}),
                  tensor(DATA_F32, {1, 4}), tensor(DATA_F32, {512})},
                 {tensor(DATA_BF16, {512, 128})}, nullptr);
    {
      const auto get_guids = reinterpret_cast<pfnGetKernelGuids>(dlsym(library, "GetKernelGuids"));
      const auto infer = reinterpret_cast<pfnGetShapeInference>(dlsym(library, "GetShapeInference"));
      if (!get_guids || !infer) throw std::runtime_error("dynamic rotary entry point missing");
      uint32_t count = 0;
      if (get_guids(DEVICE_ID_GAUDI2, &count, nullptr) != GLUE_SUCCESS)
        throw std::runtime_error("GUID count failed");
      std::vector<GuidInfo> guids(count);
      if (get_guids(DEVICE_ID_GAUDI2, &count, guids.data()) != GLUE_SUCCESS)
        throw std::runtime_error("GUID query failed");
      const GuidInfo* rotary_guid = nullptr;
      for (const auto& guid : guids)
        if (std::strcmp(guid.name, "gk_rotary_bf16_v1") == 0) rotary_guid = &guid;
      if (!rotary_guid || !rotary_guid->supportsDynamicShapes || !rotary_guid->nameHash.Value)
        throw std::runtime_error("rotary dynamic shape support is not advertised");
      auto cache = tensor(DATA_BF16, {64, 131072});
      cache.geometry.minSizes[1] = 1;
      std::vector<Tensor> inputs{tensor(DATA_BF16, {128, 4}), tensor(DATA_BF16, {64, 4}),
                                 tensor(DATA_I32, {4}), cache, tensor(DATA_I32, {1})};
      std::vector<Tensor> outputs{inputs[0], inputs[1]};
      int32_t rotary_params[]{64, 64, 1};
      HabanaKernelParams params{};
      std::strcpy(params.guid.name, rotary_guid->name);
      params.inputTensorNr = inputs.size(); params.inputTensors = inputs.data();
      params.outputTensorNr = outputs.size(); params.outputTensors = outputs.data();
      params.nodeParams.nodeParams = rotary_params; params.nodeParams.nodeParamsSize = sizeof(rotary_params);
      std::vector<TensorAccessPattern> in_patterns(inputs.size()), out_patterns(outputs.size());
      HabanaKernelInstantiation instance{};
      instance.inputTensorAccessPattern = in_patterns.data();
      instance.outputTensorAccessPattern = out_patterns.data();
      if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER)
        throw std::runtime_error("rotary rejected dynamic cache length");
      inputs[3].geometry.minSizes[1] = 0;
      if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_SIZE)
        throw std::runtime_error("rotary accepted zero cache length");
      TensorShapeInfo q_shape{}, k_shape{}, q_result{}, k_result{};
      q_shape.geometry = inputs[0].geometry; k_shape.geometry = inputs[1].geometry;
      TensorShapeInfo* shape_inputs[]{&q_shape, &k_shape, &q_shape, &q_shape, &q_shape};
      TensorShapeInfo* shape_outputs[]{&q_result, &k_result};
      ShapeInferenceParams shape_params{};
      std::strcpy(shape_params.guid.name, rotary_guid->name);
      shape_params.inputTensors = shape_inputs; shape_params.inputTensorsNr = 5;
      shape_params.outputTensorsNr = 2;
      ShapeInferenceOutput shape_result{}; shape_result.outputTensors = shape_outputs;
      if (infer(DEVICE_ID_GAUDI2, &shape_params, &shape_result) != GLUE_SUCCESS ||
          q_result.geometry.maxSizes[0] != 128 || k_result.geometry.maxSizes[0] != 64 ||
          q_result.geometry.maxSizes[1] != 4 || k_result.geometry.maxSizes[1] != 4)
        throw std::runtime_error("rotary output shape inference failed");
    }
    // Extended Track A contracts: norm weight_offset + width=576, head_major d=64/192/256,
    // online d=256, decode_normalize d=64, SWA+softcap params, paged_free bitset,
    // rotary_pos, ewise_unary, gelu_and_mul, sample_vocab.
    {
      struct { float eps; float offset; } norm2{1e-5f, 1.0f};
      const auto r576 = tensor(DATA_BF16, {576, 3});
      const auto g576 = tensor(DATA_BF16, {576});
      check_kernel(instantiate, layouts, "gk_rms_norm_bf16_v1", {r576, g576}, {r576}, &norm2,
                   false);
      // Verify 8-byte nodeParams via raw call too.
      {
        std::vector<Tensor> ins{r576, g576}, outs{r576};
        std::vector<TensorAccessPattern> ip(2), op(1);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, "gk_fused_add_rms_norm_bf16_v1");
        ins.push_back(g576);
        std::vector<Tensor> ins3{r576, r576, g576}, outs2{r576, r576};
        ip.resize(3); op.resize(2);
        p.inputTensorNr = 3; p.outputTensorNr = 2;
        p.inputTensors = ins3.data(); p.outputTensors = outs2.data();
        p.nodeParams.nodeParams = &norm2; p.nodeParams.nodeParamsSize = sizeof(norm2);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER || inst.kernel.paramsNr != 2)
          throw std::runtime_error("fused_add_rms_norm 8-byte params failed");
      }

      // kv_store_head_major width=192 (3 heads * 64) and d=256
      check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                   {tensor(DATA_BF16, {192, 4}), tensor(DATA_BF16, {192, 4}),
                    tensor(DATA_I32, {4}), tensor(DATA_I32, {3}),
                    tensor(DATA_BF16, {64, 2 * 1024 * 3})},
                   {tensor(DATA_BF16, {64, 2 * 1024 * 3})}, nullptr, true);
      check_kernel(instantiate, layouts, "gk_kv_store_head_major256_bf16_v1",
                   {tensor(DATA_BF16, {512, 4}), tensor(DATA_BF16, {512, 4}),
                    tensor(DATA_I32, {4}), tensor(DATA_I32, {3}),
                    tensor(DATA_BF16, {256, 2 * 1024 * 2})},
                   {tensor(DATA_BF16, {256, 2 * 1024 * 2})}, nullptr, true);

      // online merge/finalize width=256 + decode_normalize keys=64
      const auto data256 = tensor(DATA_F32, {256, 4, 4});
      const auto data256_bf16 = tensor(DATA_BF16, {256, 4, 4});
      check_kernel(instantiate, layouts, "gk_attn_online_merge_f32_v1",
                   {stats, stats, data256, stats, stats, data256}, {stats, stats, data256}, nullptr);
      check_kernel(instantiate, layouts, "gk_attn_online_finalize_bf16_v1",
                   {stats, data256}, {data256_bf16}, nullptr);
      check_kernel(instantiate, layouts, "gk_attn_decode_normalize_bf16_v1",
                   {tensor(DATA_F32, {64, 1, 4}), tensor(DATA_F32, {64, 4})},
                   {tensor(DATA_BF16, {64, 1, 4})}, nullptr);

      // SWA + Softcap 12-byte params
      struct { float scale; float softcap; int32_t swa; } sm{0.125f, 50.0f, 128};
      for (const char* guid : {"gk_attn_tile_softmax_bf16_v1", "gk_attn_direct_softmax_bf16_v1",
                               "gk_attn_decode_exp_f32_v1"}) {
        const bool is_tile = std::strcmp(guid, "gk_attn_tile_softmax_bf16_v1") == 0;
        const bool is_dec = std::strcmp(guid, "gk_attn_decode_exp_f32_v1") == 0;
        const auto sc = is_dec ? tensor(DATA_F32, {1024, 1, 4}) : scores;
        const auto pr = is_dec ? tensor(DATA_F32, {1024, 1, 4}) : probabilities;
        const auto pos = tensor(DATA_I32, {is_dec ? 1UL : 4UL});
        std::vector<Tensor> ins{sc, pos, tensor(DATA_I32, {3})};
        std::vector<Tensor> outs = is_tile ? std::vector<Tensor>{pr, stats, stats}
                                 : is_dec ? std::vector<Tensor>{pr, stats}
                                          : std::vector<Tensor>{pr};
        std::vector<TensorAccessPattern> ip(ins.size()), op(outs.size());
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, guid);
        p.inputTensorNr = ins.size(); p.outputTensorNr = outs.size();
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &sm; p.nodeParams.nodeParamsSize = sizeof(sm);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error(std::string(guid) + ": 12-byte softcap/SWA failed");
      }

      // paged_free with bitset + arbitrary page_size
      {
        int32_t page_size = 32;
        std::vector<Tensor> ins{tensor(DATA_I32, {2, 256}), tensor(DATA_I32, {3}),
                                tensor(DATA_I32, {1024})};
        std::vector<Tensor> outs{tensor(DATA_I32, {2, 256}), tensor(DATA_I32, {1})};
        std::vector<TensorAccessPattern> ip(3), op(2);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, "gk_paged_free_i32_v1");
        p.inputTensorNr = 3; p.outputTensorNr = 2;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &page_size; p.nodeParams.nodeParamsSize = sizeof(page_size);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error("paged_free bitset/page_size=32 failed");
      }

      // rotary_pos_bf16
      {
        auto cache = tensor(DATA_BF16, {64, 131072});
        cache.geometry.minSizes[1] = 1;
        std::vector<Tensor> ins{tensor(DATA_BF16, {128, 4}), tensor(DATA_BF16, {64, 4}),
                                tensor(DATA_I32, {4}), cache};
        std::vector<Tensor> outs{ins[0], ins[1]};
        int32_t rp[]{64, 64, 0};
        std::vector<TensorAccessPattern> ip(4), op(2);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, "gk_rotary_pos_bf16_v1");
        p.inputTensorNr = 4; p.outputTensorNr = 2;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = rp; p.nodeParams.nodeParamsSize = sizeof(rp);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error("rotary_pos_bf16 failed");
      }

      // ewise_unary_bf16 / f32
      for (const auto [guid, dt] : {std::pair{"gk_ewise_unary_bf16_v1", DATA_BF16},
                                    std::pair{"gk_ewise_unary_f32_v1", DATA_F32}}) {
        struct { int32_t op; float a; float b; } ep{6, 30.0f, 0.0f};
        std::vector<Tensor> ins{tensor(dt, {577, 5})}, outs{tensor(dt, {577, 5})};
        std::vector<TensorAccessPattern> ip(1), op(1);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, guid);
        p.inputTensorNr = 1; p.outputTensorNr = 1;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &ep; p.nodeParams.nodeParamsSize = sizeof(ep);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error(std::string(guid) + " failed");
      }

      // gelu_and_mul_bf16
      {
        check_kernel(instantiate, layouts, "gk_gelu_and_mul_bf16_v1",
                     {tensor(DATA_BF16, {1152, 4})}, {tensor(DATA_BF16, {576, 4})}, nullptr);
      }

      // sample_vocab_bf16 / f32
      for (const auto [guid, dt] : {std::pair{"gk_sample_vocab_bf16_v1", DATA_BF16},
                                    std::pair{"gk_sample_vocab_f32_v1", DATA_F32}}) {
        struct { int32_t vocab; int32_t nan_mask; } sp{1000, 1};
        const unsigned long R = 8;
        std::vector<Tensor> ins{tensor(dt, {1024, R}), tensor(DATA_F32, {R}), tensor(DATA_F32, {R}),
                                tensor(DATA_I32, {R}), tensor(DATA_F32, {R}), tensor(DATA_I32, {2, R})};
        std::vector<Tensor> outs{tensor(DATA_I32, {R})};
        std::vector<TensorAccessPattern> ip(6), op(1);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, guid);
        p.inputTensorNr = 6; p.outputTensorNr = 1;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &sp; p.nodeParams.nodeParamsSize = sizeof(sp);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error(std::string(guid) + " failed");
      }

      // 12B sliding_window=-1 on direct_softmax & decode_exp, plus 16B TileSoftmaxParameters
      {
        struct { float scale; float softcap; int32_t swa; } sm_disabled{0.125f, 0.0f, -1};
        for (const char* guid : {"gk_attn_direct_softmax_bf16_v1", "gk_attn_decode_exp_f32_v1"}) {
          const bool is_dec = std::strcmp(guid, "gk_attn_decode_exp_f32_v1") == 0;
          const auto sc = is_dec ? tensor(DATA_F32, {1024, 1, 4}) : scores;
          const auto pr = is_dec ? tensor(DATA_F32, {1024, 1, 4}) : probabilities;
          const auto pos = tensor(DATA_I32, {is_dec ? 1UL : 4UL});
          std::vector<Tensor> ins{sc, pos, tensor(DATA_I32, {3})};
          std::vector<Tensor> outs = is_dec ? std::vector<Tensor>{pr, stats} : std::vector<Tensor>{pr};
          std::vector<TensorAccessPattern> ip(ins.size()), op(outs.size());
          HabanaKernelParams p{};
          std::strcpy(p.guid.name, guid);
          p.inputTensorNr = ins.size(); p.outputTensorNr = outs.size();
          p.inputTensors = ins.data(); p.outputTensors = outs.data();
          p.nodeParams.nodeParams = &sm_disabled; p.nodeParams.nodeParamsSize = sizeof(sm_disabled);
          HabanaKernelInstantiation inst{};
          inst.inputTensorAccessPattern = ip.data();
          inst.outputTensorAccessPattern = op.data();
          if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
            throw std::runtime_error(std::string(guid) + ": 12-byte sliding_window=-1 failed");
        }
        struct { float scale; int32_t rows; float softcap; int32_t swa; } tile16{0.125f, 4, 50.0f, -1};
        std::vector<Tensor> ins{scores, tensor(DATA_I32, {4}), tensor(DATA_I32, {3})};
        std::vector<Tensor> outs{probabilities, stats, stats};
        std::vector<TensorAccessPattern> ip(3), op(3);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, "gk_attn_tile_softmax_bf16_v1");
        p.inputTensorNr = 3; p.outputTensorNr = 3;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &tile16; p.nodeParams.nodeParamsSize = sizeof(tile16);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER || inst.kernel.paramsNr != 4)
          throw std::runtime_error("gk_attn_tile_softmax_bf16_v1: 16-byte TileSoftmaxParameters failed");
      }

      // paged_alloc dynamic buckets (slots=512, pages=8, result=8192, page_size=32)
      {
        int32_t page_size = 32;
        std::vector<Tensor> ins{tensor(DATA_I32, {2, 512}), tensor(DATA_I32, {2, 8}),
                                tensor(DATA_I32, {4, 512})};
        std::vector<Tensor> outs{tensor(DATA_I32, {2, 8192})};
        std::vector<TensorAccessPattern> ip(3), op(1);
        HabanaKernelParams p{};
        std::strcpy(p.guid.name, "gk_paged_alloc_i32_v1");
        p.inputTensorNr = 3; p.outputTensorNr = 1;
        p.inputTensors = ins.data(); p.outputTensors = outs.data();
        p.nodeParams.nodeParams = &page_size; p.nodeParams.nodeParamsSize = sizeof(page_size);
        HabanaKernelInstantiation inst{};
        inst.inputTensorAccessPattern = ip.data();
        inst.outputTensorAccessPattern = op.data();
        if (instantiate(&p, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER)
          throw std::runtime_error("paged_alloc dynamic buckets failed");
      }

      // 8192-row bucket checks on kv_store_head_major & fp8_act_row_scale
      check_kernel(instantiate, layouts, "gk_kv_store_head_major_bf16_v1",
                   {tensor(DATA_BF16, {1024, 8192}), tensor(DATA_BF16, {1024, 8192}),
                    tensor(DATA_I32, {8192}), tensor(DATA_I32, {3}), head_major_pool},
                   {head_major_pool}, nullptr, true);
      check_kernel(instantiate, layouts, "gk_fp8_act_row_scale_f32_v1",
                   {tensor(DATA_BF16, {256, 8192})}, {tensor(DATA_F32, {1, 8192})}, nullptr);
    }
    dlclose(library);
    std::cout << "generic norm and attention access axes and layouts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
