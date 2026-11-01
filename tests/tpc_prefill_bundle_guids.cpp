// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
  using namespace tpc_lib_api;
  if (argc != 2) throw std::invalid_argument("expected one provider library");
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  const char* required_symbols[] = {"GetKernelGuids", "InstantiateTpcKernel", "GetShapeInference",
                                    "GetSupportedDataLayout", "GetLibVersion"};
  for (auto symbol : required_symbols)
    if (!dlsym(library, symbol)) throw std::runtime_error(std::string("missing ") + symbol);
  using GetGuids = GlueCodeReturn (*)(DeviceId, uint32_t*, GuidInfo*);
  auto get_guids = reinterpret_cast<GetGuids>(dlsym(library, "GetKernelGuids"));
  uint32_t count = 0;
  if (get_guids(DEVICE_ID_GAUDI2, &count, nullptr) != GLUE_SUCCESS || !count)
    throw std::runtime_error("provider guid count failed");
  std::vector<GuidInfo> guids(count);
  if (get_guids(DEVICE_ID_GAUDI2, &count, guids.data()) != GLUE_SUCCESS)
    throw std::runtime_error("provider guid enumeration failed");
  std::unordered_set<std::string> found;
  for (const auto& guid : guids)
    if (!found.emplace(guid.name).second) throw std::runtime_error("duplicate provider guid");
  const char* required_guids[] = {"gk_expert_prefix", "gk_expert_inverse", "gk_expert_gate",
      "gk_expert_combine", "gk_expert_decode_k8", "gk_expert_sparse_map",
      "gk_expert_counted_gather", "gk_expert_count", "gk_mimo_moe_gate_prefill_fp32_v1",
      "gk_mimo_embedding_bf16_v1", "gk_mimo_argmax_parts_bf16_v1"};
  for (auto guid : required_guids)
    if (!found.count(guid)) throw std::runtime_error(std::string("missing ") + guid);
  std::printf("prefill provider: %u unique guids, all required entries present\n", count);
  dlclose(library);
}
