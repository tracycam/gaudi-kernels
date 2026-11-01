// SPDX-License-Identifier: Apache-2.0
// Check actual split-tile device routing, including adversarial distributions.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;
namespace {
#ifndef TEST_ROWS
#define TEST_ROWS 512
#endif
constexpr int kRows = TEST_ROWS, kRoutes = 8, kExperts = 384, kLength = kRows * kRoutes;
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("partition routes: Synapse failure " + std::to_string(status));
}
struct Tensor {
  std::string name;
  std::vector<uint64_t> shape;
  synSectionHandle section = nullptr;
  synTensor tensor = nullptr;
  synLaunchTensorInfo info{};
  uint64_t bytes = 4;
};
}

int main() {
  DeviceRuntime::Config config{};
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  check(synGraphCreate(&graph, synDeviceGaudi2));
  std::vector<Tensor> tensors;
  tensors.reserve(17);
  auto make = [&](const char* name, std::initializer_list<uint64_t> shape) -> size_t {
    tensors.emplace_back();
    auto& t = tensors.back();
    t.name = name; t.shape = shape;
    check(synSectionCreate(&t.section, 0, graph));
    check(synSectionSetPersistent(t.section, true));
    synTensorDescriptor desc{};
    desc.m_name = t.name.c_str(); desc.m_dataType = syn_type_int32; desc.m_dims = t.shape.size();
    for (size_t d = 0; d < t.shape.size(); ++d) {
      desc.m_sizes[d] = desc.m_minSizes[d] = t.shape[d];
      t.info.tensorSize[d] = t.shape[d]; t.bytes *= t.shape[d];
    }
    check(synTensorCreate(&t.tensor, &desc, t.section, 0));
    t.info.tensorName = t.name.c_str(); t.info.tensorType = DATA_TENSOR;
    return tensors.size() - 1;
  };
  auto node = [&](const char* guid, const char* name, std::initializer_list<size_t> in,
                  std::initializer_list<size_t> out, const int32_t* p, size_t n) {
    std::vector<synTensor> x, y;
    for (auto i : in) x.push_back(tensors[i].tensor);
    for (auto i : out) y.push_back(tensors[i].tensor);
    check(synNodeCreate(graph, x.data(), y.data(), x.size(), y.size(), p, n * sizeof(*p), guid, name, nullptr, nullptr));
  };
  const auto ids = make("ids", {kRoutes, kRows});
  const auto flat = make("flat", {kLength});
  const auto counts = make("counts", {kExperts});
  const auto flags = make("flags", {kRows});
  const auto offsets = make("offsets", {kLength / 64 + 1, kExperts});
  const int32_t expert_param = kExperts;
  node("gk_expert_count", "count", {ids, flat}, {counts, flags, offsets}, &expert_param, 1);
  struct Band { int rows, slots; size_t experts, valid, status, inverse; };
  std::array<Band, 1> bands{};
  {
    const int rows = kRows / 8, slots = kExperts + (kLength + rows - 1) / rows;
    auto pref = make("prefix", {kExperts + 1});
    auto experts = make("experts", {static_cast<uint64_t>(slots)});
    auto base = make("base", {static_cast<uint64_t>(slots)});
    auto valid = make("valid", {static_cast<uint64_t>(slots)});
    auto status = make("status", {1});
    auto inverse = make("inverse", {kLength});
    const int32_t prefix_params[] = {kRoutes, rows, slots};
    const int32_t inverse_params[] = {rows, slots};
    node("gk_expert_prefix_split", "prefix_node", {counts, flags}, {pref, experts, base, valid, status}, prefix_params, 3);
    node("gk_expert_inverse", "inverse_node", {ids, flat, pref, status, offsets}, {inverse}, inverse_params, 2);
    bands[0] = {rows, slots, experts, valid, status, inverse};
  }
  check(synGraphCompile(&recipe, graph, "partition_routes_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  std::vector<synLaunchTensorInfo> bindings;
  for (auto& t : tensors) { t.info.pTensorAddress = arena.allocate(t.bytes); bindings.push_back(t.info); }
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto& t : tensors) check(synSectionDestroy(t.section));
  PreparedLaunch launch{recipe, arena.ensure_workspace(workspace), bindings, 0};
  launch.validate();
  auto read = [&](size_t i, std::vector<int32_t>& dst) {
    dst.resize(tensors[i].bytes / sizeof(int32_t));
    arena.copy_d2h_sync(runtime.d2h_stream(), bindings[i].pTensorAddress, dst.data(), tensors[i].bytes);
  };
  for (int scenario = 0; scenario < 4; ++scenario) {
    std::vector<int32_t> route(kLength);
    for (int t = 0; t < kRows; ++t) for (int r = 0; r < kRoutes; ++r) {
      route[t * kRoutes + r] = scenario == 0 ? r :
          scenario == 1 ? (r == 0 ? 0 : 1 + ((t * 7 + r - 1) % 383)) : scenario == 2 ? (t * kRoutes + r) % kExperts :
          (t < kRows / 2 ? r : 8 + ((t * kRoutes + r) % (kExperts - 8)));
    }
    for (auto i : {ids, flat}) arena.copy_h2d_sync(runtime.h2d_stream(), route.data(), bindings[i].pTensorAddress, tensors[i].bytes);
    launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
    std::vector<int> ownership(kLength);
    for (const auto& b : bands) {
      std::vector<int32_t> experts, valid, status, inverse;
      read(b.experts, experts); read(b.valid, valid); read(b.status, status); read(b.inverse, inverse);
      if (status[0] != 0) throw std::runtime_error("partition status " + std::to_string(status[0]));
      std::vector<int> seen(b.slots * b.rows);
      for (int q = 0; q < kLength; ++q) {
        int pos = inverse[q];
        if (pos < 0) continue;
        if (pos >= b.slots * b.rows || experts[pos / b.rows] != route[q] ||
            pos % b.rows >= valid[pos / b.rows] || ++seen[pos] != 1)
          throw std::runtime_error("partition mapping mismatch at route " + std::to_string(q));
        ++ownership[q];
      }
    }
    for (int n : ownership) if (n != 1) throw std::runtime_error("route missing or duplicated");
    std::printf("scenario %d: status=0, %d routes owned exactly once\n", scenario, kLength);
  }
  check(synRecipeDestroy(recipe));
}
