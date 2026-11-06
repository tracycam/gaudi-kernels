// Verify the generic provider exposes only the supported dense primitives.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>
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
    for (const auto& guid : guids) names.emplace(guid.name);
    const std::set<std::string> expected{
        "gk_add_bf16_v1", "gk_fused_add_rms_norm_bf16_v1", "gk_rms_norm_bf16_v1",
        "gk_silu_and_mul_bf16_v1", "gk_kv_store_bf16_v1", "gk_rotary_bf16_v1",
        "gk_paged_metadata_i32_v1", "gk_token_indices_i32_v1", "gk_cast_f32_to_bf16_v1",
        "gk_index_put_i64_v1", "gk_scheduler_metadata_i32_v1", "gk_scheduler_tokens_i32_v1",
        "gk_attn_tile_softmax_bf16_v1", "gk_attn_online_merge_f32_v1", "gk_attn_online_finalize_bf16_v1"};
    if (count != expected.size() || names != expected)
      throw std::runtime_error("unexpected provider GUID set");
    dlclose(library);
    std::cout << "generic provider exposes 15 expected GUIDs; old attention GUIDs absent\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
