// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <algorithm>
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
  if (status != synSuccess) throw std::runtime_error("MoE gate test: Synapse failure");
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
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 3> tensors{};
  std::array<synLaunchTensorInfo, 3> bindings{};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  const char* names[] = {"partial", "ids", "activation"};
  const synDataType types[] = {syn_type_float, syn_type_int32, syn_type_bf16};
  const uint64_t widths[] = {12288, 8, 256};
  const uint64_t rows[] = {1, 1, 8};
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i]; desc.m_dataType = types[i]; desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = widths[i];
    desc.m_sizes[1] = desc.m_minSizes[1] = rows[i];
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    bindings[i].tensorName = names[i]; bindings[i].tensorType = DATA_TENSOR;
    bindings[i].tensorSize[0] = widths[i]; bindings[i].tensorSize[1] = rows[i];
  }
  const int32_t splits = 3;
  check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, &splits, sizeof(splits),
                      "gk_mimo_moe_gate_fp32_v1", "fp32_gate", nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, "moe_fp32_gate_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto section : sections) check(synSectionDestroy(section));

  std::vector<float> partial(12288);
  uint32_t random = 0x6c8e9cf5;
  for (auto& value : partial) {
    random ^= random << 13; random ^= random >> 17; random ^= random << 5;
    value = (static_cast<int32_t>(random % 20001) - 10000) / 4096.0f;
  }
  // Include the observed input and neighboring range-reduction boundaries.
  std::array<float, 750> boundaries{};
  size_t boundary_count = 0;
  const float ln2 = std::log(2.0f);
  for (int multiple = 1; multiple <= 125; ++multiple) {
    for (float sign : {-1.0f, 1.0f}) {
      const float value = sign * static_cast<float>(multiple) * ln2;
      boundaries[boundary_count++] = std::nextafter(value, -std::numeric_limits<float>::infinity());
      boundaries[boundary_count++] = value;
      boundaries[boundary_count++] = std::nextafter(value, std::numeric_limits<float>::infinity());
    }
  }
  for (size_t index = 0; index < boundary_count; ++index) {
    const int slot = index / 256;
    const int row = index % 256;
    const int packed_row = (row / 128) * 128 + (row % 128) / 2 + (row % 2) * 64;
    const int gate_base = slot * 3 * 512 + packed_row;
    for (int split = 0; split < 3; ++split) {
      partial[gate_base + split * 512] = split == 0 ? boundaries[index] : 0.0f;
      partial[gate_base + 256 + split * 512] = split == 0 ? 0.2160845696926117f : 0.0f;
    }
  }
  std::array<int32_t, 8> ids{1, 7, 19, 31, 47, 99, 150, 383};
  std::vector<uint16_t> actual(8 * 256);
  bindings[0].pTensorAddress = arena.allocate(partial.size() * sizeof(float));
  bindings[1].pTensorAddress = arena.allocate(sizeof(ids));
  bindings[2].pTensorAddress = arena.allocate(actual.size() * sizeof(uint16_t));
  arena.copy_h2d_sync(runtime.h2d_stream(), partial.data(), bindings[0].pTensorAddress,
                      partial.size() * sizeof(float));
  arena.copy_h2d_sync(runtime.h2d_stream(), ids.data(), bindings[1].pTensorAddress, sizeof(ids));
  PreparedLaunch launch{recipe, arena.ensure_workspace(workspace),
                        {bindings.begin(), bindings.end()}, 0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[2].pTensorAddress,
                      actual.data(), actual.size() * sizeof(uint16_t));
  double error_sq = 0, expected_sq = 0;
  float max_abs = 0;
  unsigned boundary_failures = 0;
  for (int slot = 0; slot < 8; ++slot) {
    for (int row = 0; row < 256; ++row) {
      // The MME partial stores even lanes followed by odd lanes in each block.
      const int packed_row = (row / 128) * 128 + (row % 128) / 2 + (row % 2) * 64;
      const int gr = slot * 512 + packed_row, ur = gr + 256;
      const int gb = (gr / 512) * 1536 + gr % 512;
      const int ub = (ur / 512) * 1536 + ur % 512;
      float gate = 0, up = 0;
      for (int split = 0; split < 3; ++split) {
        gate += partial[gb + split * 512]; up += partial[ub + split * 512];
      }
      const double expected = widen(bf16(static_cast<float>(
          static_cast<double>(gate) / (1.0 + std::exp(-static_cast<double>(gate))) * up)));
      const double difference = widen(actual[slot * 256 + row]) - expected;
      if (static_cast<size_t>(slot * 256 + row) < boundary_count && difference != 0) {
        std::printf("boundary gate=%.9g actual=%.9g expected=%.9g\n", gate,
                    widen(actual[slot * 256 + row]), expected);
        ++boundary_failures;
      }
      error_sq += difference * difference; expected_sq += expected * expected;
      max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
    }
  }
  const double relative = std::sqrt(error_sq / expected_sq);
  std::printf("FP32 MoE gate relative_l2=%.9g max_abs=%.9g\n", relative, max_abs);
  check(synRecipeDestroy(recipe));
  if (boundary_failures) throw std::runtime_error("FP32 MoE gate fails boundary inputs");
  if (!(relative <= 1e-3)) throw std::runtime_error("FP32 MoE gate exceeds mathematical tolerance");
}
