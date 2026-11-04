// SPDX-License-Identifier: Apache-2.0
// Standalone Synapse test: the kernel consumes BF16 and rounds an FP32 sum.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <synapse_common_types.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}

uint16_t bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows) {
  const size_t count = static_cast<size_t>(width) * rows;
  std::vector<uint16_t> left(count), right(count), actual(count);
  for (size_t i = 0; i < count; ++i) {
    const auto a = static_cast<int32_t>((i * 48271ULL + 17) % 200001) - 100000;
    const auto b = static_cast<int32_t>((i * 69621ULL + 37) % 200001) - 100000;
    left[i] = bf16(std::ldexp(static_cast<float>(a), static_cast<int>((i / 127) % 16) - 20));
    right[i] = bf16(std::ldexp(static_cast<float>(b), static_cast<int>((i / 131) % 16) - 20));
  }
  arena.reset();
  const std::array<uint64_t, 3> addresses{arena.allocate(count * 2), arena.allocate(count * 2),
                                          arena.allocate(count * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), left.data(), addresses[0], count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), right.data(), addresses[1], count * 2);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 3> tensors{};
  std::array<synSectionHandle, 3> sections{};
  std::array<synLaunchTensorInfo, 3> launch{};
  const std::string prefix = "generic_add_w" + std::to_string(width) + "_r" + std::to_string(rows);
  const std::array<std::string, 3> names{prefix + "_t0", prefix + "_t1", prefix + "_t2"};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_bf16;
    desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = width;
    desc.m_sizes[1] = desc.m_minSizes[1] = rows;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = width;
    launch[i].tensorSize[1] = rows;
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, nullptr, 0,
                      "gk_add_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), addresses[2], actual.data(), count * 2);
  size_t mismatches = 0;
  for (size_t i = 0; i < count; ++i) mismatches += actual[i] != bf16(widen(left[i]) + widen(right[i]));
  std::cout << "width " << width << " rows " << rows << " bit_differences " << mismatches << '\n';
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  if (mismatches) throw std::runtime_error("BF16 add mismatch");
}
}  // namespace

int main() {
  try {
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 128ULL * 1024 * 1024);
    for (const auto [width, rows] : {std::pair{1, 1}, {7, 3}, {127, 7}, {128, 1},
                                    {129, 7}, {7168, 7}, {22016, 137}})
      run(runtime, arena, width, rows);
    std::cout << "standalone BF16 add passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
