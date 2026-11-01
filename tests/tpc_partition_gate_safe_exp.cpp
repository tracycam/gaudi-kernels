// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("partition gate: Synapse failure");
}
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}
float widen(uint16_t bits) {
  const uint32_t raw = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}
}

int main() {
  using namespace gaudirt::runtime;
  DeviceRuntime::Config config{};
  config.module_id = 0;
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synSectionHandle, 4> sections{};
  std::array<synTensor, 4> tensors{};
  std::array<synLaunchTensorInfo, 4> bindings{};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  const char* names[] = {"partial", "valid", "status", "activation"};
  const synDataType types[] = {syn_type_float, syn_type_int32, syn_type_int32, syn_type_bf16};
  const unsigned dims[] = {3, 1, 1, 3};
  const uint64_t shapes[][3] = {{512, 3, 1}, {1, 1, 1}, {1, 1, 1}, {256, 3, 1}};
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i]; desc.m_dataType = types[i]; desc.m_dims = dims[i];
    for (unsigned d = 0; d < dims[i]; ++d) {
      desc.m_sizes[d] = desc.m_minSizes[d] = shapes[i][d];
      bindings[i].tensorSize[d] = shapes[i][d];
    }
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    bindings[i].tensorName = names[i]; bindings[i].tensorType = DATA_TENSOR;
  }
  const int32_t slot = 0;
  check(synNodeCreate(graph, tensors.data(), &tensors[3], 3, 1, &slot, sizeof(slot),
                      "gk_expert_gate", "partition_safe_gate", nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, "partition_safe_gate_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto section : sections) check(synSectionDestroy(section));

  std::array<float, 750> boundaries{};
  size_t n = 0;
  const float ln2 = std::log(2.0f);
  for (int multiple = 1; multiple <= 125; ++multiple) {
    for (float sign : {-1.0f, 1.0f}) {
      float value = sign * static_cast<float>(multiple) * ln2;
      boundaries[n++] = std::nextafter(value, -std::numeric_limits<float>::infinity());
      boundaries[n++] = value;
      boundaries[n++] = std::nextafter(value, std::numeric_limits<float>::infinity());
    }
  }
  std::vector<float> partial(512 * 3, 0.0f);
  for (size_t i = 0; i < 3 * 256; ++i) {
    partial[(i / 256) * 512 + 256 + i % 256] = static_cast<float>(1 + i % 7) / 8.0f;
  }
  for (size_t i = 0; i < boundaries.size(); ++i) {
    partial[(i / 256) * 512 + i % 256] = boundaries[i];
  }
  const int32_t valid = 3, status = 0;
  std::vector<uint16_t> actual(256 * 3);
  bindings[0].pTensorAddress = arena.allocate(partial.size() * sizeof(float));
  bindings[1].pTensorAddress = arena.allocate(sizeof(valid));
  bindings[2].pTensorAddress = arena.allocate(sizeof(status));
  bindings[3].pTensorAddress = arena.allocate(actual.size() * sizeof(uint16_t));
  arena.copy_h2d_sync(runtime.h2d_stream(), partial.data(), bindings[0].pTensorAddress, partial.size() * sizeof(float));
  arena.copy_h2d_sync(runtime.h2d_stream(), &valid, bindings[1].pTensorAddress, sizeof(valid));
  arena.copy_h2d_sync(runtime.h2d_stream(), &status, bindings[2].pTensorAddress, sizeof(status));
  PreparedLaunch launch{recipe, arena.ensure_workspace(workspace),
                        {bindings.begin(), bindings.end()}, 0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[3].pTensorAddress, actual.data(), actual.size() * sizeof(uint16_t));
  unsigned failures = 0;
  for (size_t i = 0; i < boundaries.size(); ++i) {
    const double gate = boundaries[i];
    const float silu = widen(bf16(static_cast<float>(gate / (1.0 + std::exp(-gate)))));
    const float up = static_cast<float>(1 + i % 7) / 8.0f;
    const auto expected = bf16(silu * up);
    if (actual[i] != expected) {
      if (failures < 8) std::printf("boundary %.9g actual %.9g expected %.9g\n", gate, widen(actual[i]), widen(expected));
      ++failures;
    }
  }
  check(synRecipeDestroy(recipe));
  std::printf("partition gate: %zu boundaries, %u mismatches\n", boundaries.size(), failures);
  if (failures) throw std::runtime_error("partition gate boundary mismatch");
}
