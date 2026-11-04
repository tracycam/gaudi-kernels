// SPDX-License-Identifier: Apache-2.0
// Verify physical KV slot writes and untouched bytes for runtime valid_rows.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <synapse_common_types.hpp>

#include <array>
#include <bit>
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

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows) {
  const uint32_t bucket = std::bit_ceil(rows);
  const uint32_t slots = 2 * bucket + 5;
  const size_t source_count = static_cast<size_t>(width) * bucket;
  const size_t pool_count = static_cast<size_t>(width) * slots;
  constexpr uint16_t key_sentinel = 0x3e80, value_sentinel = 0x3f00;
  std::vector<uint16_t> key(source_count), value(source_count), key_pool(pool_count, key_sentinel),
      value_pool(pool_count, value_sentinel), actual_key(pool_count), actual_value(pool_count);
  std::vector<int32_t> locations(bucket, 0);
  std::vector<int32_t> source_for_slot(slots, -1);
  for (uint32_t row = 0; row < bucket; ++row) {
    if (row < rows) {
      locations[row] = 1 + 2 * ((row * 199) % bucket);
      source_for_slot[locations[row]] = row;
    }
    for (uint32_t col = 0; col < width; ++col) {
      const auto i = static_cast<size_t>(row) * width + col;
      key[i] = static_cast<uint16_t>(0x3f80 + (i % 127));
      value[i] = static_cast<uint16_t>(0x4000 + ((i * 7) % 127));
    }
  }
  arena.reset();
  const std::array<uint64_t, 6> addresses{
      arena.allocate(source_count * 2), arena.allocate(source_count * 2), arena.allocate(bucket * 4),
      arena.allocate(sizeof(int32_t)), arena.allocate(pool_count * 2), arena.allocate(pool_count * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), key.data(), addresses[0], source_count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), value.data(), addresses[1], source_count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), locations.data(), addresses[2], bucket * 4);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 6> tensors{};
  std::array<synSectionHandle, 6> sections{};
  std::array<synLaunchTensorInfo, 6> launch{};
  const std::string prefix = "generic_kv_store_w" + std::to_string(width) + "_b" + std::to_string(bucket);
  std::array<std::string, 6> names;
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = i == 2 || i == 3 ? syn_type_int32 : syn_type_bf16;
    desc.m_dims = i == 2 || i == 3 ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i == 2 ? bucket : i == 3 ? 1 : width;
    if (desc.m_dims == 2) desc.m_sizes[1] = desc.m_minSizes[1] = i < 2 ? bucket : slots;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = desc.m_sizes[0];
    if (desc.m_dims == 2) launch[i].tensorSize[1] = desc.m_sizes[1];
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), &tensors[4], 4, 2, nullptr, 0,
                      "gk_kv_store_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  for (const uint32_t valid : {rows, uint32_t{1}}) {
    const int32_t metadata = valid;
    arena.copy_h2d_sync(runtime.h2d_stream(), &metadata, addresses[3], sizeof(metadata));
    arena.copy_h2d_sync(runtime.h2d_stream(), key_pool.data(), addresses[4], pool_count * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), value_pool.data(), addresses[5], pool_count * 2);
    check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                    arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_key.data(), pool_count * 2);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[5], actual_value.data(), pool_count * 2);
    size_t wrong_targets = 0, touched_other = 0;
    for (uint32_t slot = 0; slot < slots; ++slot) {
      const bool targeted = source_for_slot[slot] >= 0 && static_cast<uint32_t>(source_for_slot[slot]) < valid;
      const uint32_t row = targeted ? source_for_slot[slot] : 0;
      for (uint32_t col = 0; col < width; ++col) {
        const auto dst = static_cast<size_t>(slot) * width + col;
        const auto src = static_cast<size_t>(row) * width + col;
        if (targeted) {
          wrong_targets += actual_key[dst] != key[src] || actual_value[dst] != value[src];
        } else {
          touched_other += actual_key[dst] != key_sentinel || actual_value[dst] != value_sentinel;
        }
      }
    }
    std::cout << "width " << width << " bucket " << bucket << " valid_rows " << valid
              << " incorrect_target_elements " << wrong_targets
              << " touched_other_elements " << touched_other << std::endl;
    if (wrong_targets || touched_other) throw std::runtime_error("KV slot write mismatch or tail pollution");
  }
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
}
}  // namespace

int main() {
  try {
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 128ULL * 1024 * 1024);
    for (const auto width : {512u, 1024u})
      for (const auto rows : {1u, 7u, 33u, 137u, 1025u}) run(runtime, arena, width, rows);
    run(runtime, arena, 17, 7);
    std::cout << "KV valid_rows and untouched-slot gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
