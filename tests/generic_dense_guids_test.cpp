// Verify the generic provider exposes only the supported dense primitives.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using namespace tpc_lib_api;
  try {
    if (argc != 2) throw std::invalid_argument("expected the provider shared library path");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    using GetGuids = GlueCodeReturn (*)(DeviceId, uint32_t*, GuidInfo*);
    const auto get = reinterpret_cast<GetGuids>(dlsym(library, "GetKernelGuids"));
    if (!get) throw std::runtime_error("GetKernelGuids is missing");
    uint32_t count = 0;
    if (get(DEVICE_ID_GAUDI2, &count, nullptr) != GLUE_SUCCESS)
      throw std::runtime_error("provider count failed");
    std::vector<GuidInfo> guids(count);
    if (get(DEVICE_ID_GAUDI2, &count, guids.data()) != GLUE_SUCCESS)
      throw std::runtime_error("provider enumeration failed");
    std::set<std::string> names;
    for (const auto& guid : guids) {
      names.emplace(guid.name);
      if (std::strcmp(guid.name, "gk_attn_tile_indices_i32_v1") == 0 && !guid.supportsDynamicShapes)
        throw std::runtime_error("tile-index kernel did not advertise dynamic input shapes");
    }
    const std::set<std::string> expected{
        "gk_add_bf16_v1", "gk_fused_add_rms_norm_bf16_v1", "gk_rms_norm_bf16_v1",
        "gk_silu_and_mul_bf16_v1", "gk_kv_store_bf16_v1", "gk_rotary_bf16_v1",
        "gk_paged_metadata_i32_v1", "gk_token_indices_i32_v1", "gk_cast_f32_to_bf16_v1",
        "gk_index_put_i64_v1", "gk_scheduler_metadata_i32_v1", "gk_scheduler_tokens_i32_v1",
        "gk_attn_tile_softmax_bf16_v1", "gk_attn_online_merge_f32_v1", "gk_attn_online_finalize_bf16_v1",
        "gk_kv_store_head_major_bf16_v1", "gk_attn_decode_exp_f32_v1",
        "gk_attn_decode_normalize_bf16_v1", "gk_attn_direct_softmax_bf16_v1",
        "gk_attn_tile_indices_i32_v1", "gk_attn_pack_stats_f32_v1",
        "gk_paged_alloc_i32_v1", "gk_paged_free_i32_v1",
        "gk_argmax_parts_bf16_v1", "gk_argmax_parts_f32_v1", "gk_argmax_finish_i32_v1",
        "gk_fp8_block_quant_bf16_v1", "gk_fp8_block_reduce_bf16_v1",
        "gk_fp8_token_quant_bf16_v1", "gk_fp8_channel_finish_bf16_v1",
        "gk_fp8_act_row_scale_f32_v1", "gk_fp8_weight_blk_scale_f32_v1",
        "gk_fp8_quant_act_fp16_1k_v1", "gk_fp8_quant_act_fp8_2k_v1",
        "gk_fp8_dequant_weight_fp16_1k_v1", "gk_fp8_dequant_weight_fp8_2k_v1",
        "gk_fp8_hybrid_epilogue_bf16_v1", "gk_fp8_hybrid_epilogue_bias_bf16_v1"};
    if (count != expected.size() || names != expected)
      throw std::runtime_error("unexpected provider GUID set");

    using Instantiate = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);
    const auto instantiate = reinterpret_cast<Instantiate>(dlsym(library, "InstantiateTpcKernel"));
    if (!instantiate) throw std::runtime_error("InstantiateTpcKernel is missing");
    Tensor inputs[4]{}, outputs[2]{};
    const auto shape = [](Tensor& tensor, TensorDataType type, uint64_t width, uint64_t rows) {
      tensor.geometry.dataType = type;
      tensor.geometry.dims = rows ? 2 : 1;
      tensor.geometry.maxSizes[0] = tensor.geometry.minSizes[0] = width;
      if (rows) tensor.geometry.maxSizes[1] = tensor.geometry.minSizes[1] = rows;
    };
    shape(inputs[2], DATA_I32, 8, 0);
    shape(inputs[3], DATA_I32, 1, 0);
    HabanaKernelParams params{};
    std::strcpy(params.guid.name, "gk_kv_store_bf16_v1");
    params.inputTensorNr = 4;
    params.inputTensors = inputs;
    params.outputTensorNr = 2;
    params.outputTensors = outputs;
    for (const uint64_t width : {17u, 64u, 128u, 129u, 512u, 1024u}) {
      shape(inputs[0], DATA_BF16, width, 8);
      shape(inputs[1], DATA_BF16, width, 8);
      shape(outputs[0], DATA_BF16, width, 32);
      shape(outputs[1], DATA_BF16, width, 32);
      TensorAccessPattern in_patterns[4]{}, out_patterns[2]{};
      HabanaKernelInstantiation instance{};
      instance.inputTensorAccessPattern = in_patterns;
      instance.outputTensorAccessPattern = out_patterns;
      const auto status = instantiate(&params, &instance);
      if (width % 128) {
        if (status != GLUE_INCOMPATIBLE_INPUT_SIZE)
          throw std::runtime_error("KV store accepted a non-aligned width");
        continue;
      }
      if (status != GLUE_INSUFFICIENT_ELF_BUFFER || !instance.kernel.elfSize ||
          instance.indexSpaceRank != 2 || instance.indexSpaceGeometry[0] != width / 128 ||
          instance.indexSpaceGeometry[1] != 8)
        throw std::runtime_error("KV store rejected an aligned width or mapped it incorrectly");
      std::vector<uint8_t> elf(instance.kernel.elfSize);
      instance.kernel.kernelElf = elf.data();
      if (instantiate(&params, &instance) != GLUE_SUCCESS)
        throw std::runtime_error("KV store ELF instantiation failed");
    }
    dlclose(library);
    std::cout << "generic provider GUIDs and KV store width contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
