// SPDX-License-Identifier: Apache-2.0
// Device validation of token IDs and the inactive static-bucket tail.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}

void run(DeviceRuntime& runtime, HbmArena& arena, int rows, bool wide) {
  constexpr int32_t vocabulary = 32000;
  const int bucket = std::bit_ceil(static_cast<unsigned>(rows)), lanes = wide ? 2 : 1;
  const std::array<size_t, 4> bytes{static_cast<size_t>(bucket * lanes * 4), 4,
                                    static_cast<size_t>(bucket * 4), static_cast<size_t>(bucket * 4)};
  arena.reset();
  std::array<uint64_t, 4> addresses{};
  for (size_t i = 0; i < addresses.size(); ++i) addresses[i] = arena.allocate(bytes[i]);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 4> tensors{};
  std::array<synSectionHandle, 4> sections{};
  std::array<synLaunchTensorInfo, 4> launch{};
  std::array<std::string, 4> names;
  const std::string prefix = "token_indices_b" + std::to_string(bucket) + (wide ? "_wide" : "_narrow");
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_int32;
    desc.m_dims = i == 0 ? 2 : 1;
    desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 ? lanes : i == 1 ? 1 : bucket;
    if (i == 0) desc.m_sizes[1] = desc.m_minSizes[1] = bucket;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    for (unsigned dim = 0; dim < desc.m_dims; ++dim) launch[i].tensorSize[dim] = desc.m_sizes[dim];
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 2, 2, 2, &vocabulary, sizeof(vocabulary),
                      "gk_token_indices_i32_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());

  std::vector<int32_t> input(bucket * lanes, -1), result(bucket), status(bucket);
  for (int row = 0; row < rows; ++row) {
    input[row * lanes] = (row * 73) % vocabulary;
    if (wide) input[row * lanes + 1] = 0;
  }
  const auto valid_input = input;
  auto scenario = [&](int invalid) {
    input = valid_input;
    if (invalid == 1) input[0] = -1;
    if (invalid == 2) input[0] = vocabulary;
    if (invalid == 3) input[1] = 1;
    arena.copy_h2d_sync(runtime.h2d_stream(), input.data(), addresses[0], bytes[0]);
    arena.copy_h2d_sync(runtime.h2d_stream(), &rows, addresses[1], bytes[1]);
    check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(), arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[2], result.data(), bytes[2]);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[3], status.data(), bytes[3]);
    for (int row = 0; row < bucket; ++row) {
      const bool bad = invalid != 0 && row == 0;
      const int expected = row < rows && !bad ? input[row * lanes] : 0;
      if (result[row] != expected || status[row] != static_cast<int>(bad))
        throw std::runtime_error("token validation mismatch at row " + std::to_string(row));
    }
  };
  scenario(0);
  scenario(1);
  scenario(2);
  if (wide) scenario(3);
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  std::cout << "rows=" << rows << " bucket=" << bucket << " lanes=" << lanes << " passed\n";
}
}  // namespace

int main() {
  try {
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 32ULL * 1024 * 1024);
    for (int rows : {1, 3, 7, 137, 1025, 4096})
      for (bool wide : {false, true}) run(runtime, arena, rows, wide);
    std::cout << "token validation and zero tail passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
