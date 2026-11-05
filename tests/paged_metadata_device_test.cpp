// SPDX-License-Identifier: Apache-2.0
// Validate mixed-varlen scheduler metadata without reading device KV data on the host.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <synapse_common_types.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}

void run(DeviceRuntime& runtime, HbmArena& arena, bool wide) {
  constexpr int bucket = 8, context = 16, request_count = 4, slots = 65;
  std::vector<int32_t> base_locations(bucket, 0), base_requests(bucket, 0), base_positions(bucket, 0);
  std::vector<int32_t> base_mapping(request_count * context), base_info(1 + 4 * bucket, 0);
  for (int req = 0; req < request_count; ++req)
    for (int pos = 0; pos < context; ++pos) base_mapping[req * context + pos] = 1 + req * context + pos;
  base_info[0] = 3;
  const std::array<int, 3> reqs{3, 1, 2}, prefixes{2, 1, 3}, lengths{2, 3, 2};
  int cursor = 0;
  for (int batch = 0; batch < 3; ++batch) {
    base_requests[batch] = reqs[batch];
    base_info[1 + 2 * bucket + batch] = prefixes[batch];
    base_info[1 + 3 * bucket + batch] = lengths[batch];
    for (int i = 0; i < lengths[batch]; ++i, ++cursor) {
      const int position = prefixes[batch] + i;
      base_info[1 + cursor] = batch;
      base_info[1 + bucket + cursor] = position;
      base_positions[cursor] = position;
      base_locations[cursor] = base_mapping[reqs[batch] * context + position];
    }
  }
  const int lanes = wide ? 2 : 1;
  std::array<uint64_t, 10> addresses{};
  const std::array<size_t, 10> sizes{bucket * 4u * lanes, bucket * 4u * lanes, bucket * 4u * lanes,
                                     request_count * context * 4, 4, (1 + 4 * bucket) * 4,
                                     bucket * 4, bucket * 4, bucket * 4, bucket * 4};
  arena.reset();
  for (size_t i = 0; i < sizes.size(); ++i) addresses[i] = arena.allocate(sizes[i]);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 10> tensors{};
  std::array<synSectionHandle, 10> sections{};
  std::array<synLaunchTensorInfo, 10> launch{};
  std::array<std::string, 10> names;
  const std::string prefix = wide ? "generic_paged_metadata_wide" : "generic_paged_metadata_narrow";
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_int32;
    desc.m_dims = i <= 3 ? 2 : 1;
    desc.m_sizes[0] = desc.m_minSizes[0] = i < 3 ? lanes : i == 3 ? context :
        i == 4 ? 1 : i == 5 ? 1 + 4 * bucket : bucket;
    if (desc.m_dims == 2) desc.m_sizes[1] = desc.m_minSizes[1] = i == 3 ? request_count : bucket;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    for (unsigned dim = 0; dim < desc.m_dims; ++dim) launch[i].tensorSize[dim] = desc.m_sizes[dim];
  }
  const int32_t params = slots;
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 6, 6, 4, &params, sizeof(params),
                      "gk_paged_metadata_i32_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());

  auto scenario = [&](const char* label, std::vector<int32_t> locations, std::vector<int32_t> requests,
                      std::vector<int32_t> positions, std::vector<int32_t> mapping,
                      std::vector<int32_t> info, int32_t valid, int expected_error, int high_input = -1) {
    const auto pack = [&](const std::vector<int32_t>& values) {
      std::vector<int32_t> packed(bucket * lanes, 0);
      for (int i = 0; i < bucket; ++i) packed[i * lanes] = values[i];
      return packed;
    };
    auto packed_locations = pack(locations), packed_requests = pack(requests), packed_positions = pack(positions);
    if (high_input >= 0) {
      if (!wide) throw std::runtime_error("high word test requires int64 input");
      if (high_input == 0) packed_locations[1] = 1;
      if (high_input == 1) packed_requests[1] = 1;
      if (high_input == 2) packed_positions[1] = 1;
    }
    const std::array<const void*, 6> inputs{packed_locations.data(), packed_requests.data(), packed_positions.data(),
                                            mapping.data(), &valid, info.data()};
    for (size_t i = 0; i < inputs.size(); ++i)
      arena.copy_h2d_sync(runtime.h2d_stream(), inputs[i], addresses[i], sizes[i]);
    check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(), arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    std::vector<int32_t> row_locations(bucket, -9), expanded(bucket, -9), row_positions(bucket, -9), status(bucket, -9);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[6], row_locations.data(), sizes[6]);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[7], expanded.data(), sizes[7]);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[8], row_positions.data(), sizes[8]);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[9], status.data(), sizes[9]);
    if (!expected_error) {
      if (std::any_of(status.begin(), status.end(), [](int value) { return value != 0; }))
        throw std::runtime_error(std::string(label) + ": valid metadata rejected");
      for (int row = 0; row < valid; ++row)
        if (expanded[row] != requests[info[1 + row]] || row_locations[row] != locations[row] ||
            row_positions[row] != positions[row])
          throw std::runtime_error(std::string(label) + ": wrong expanded request");
      for (int row = valid; row < bucket; ++row)
        if (expanded[row] != 0 || row_locations[row] != 0 || row_positions[row] != 0)
          throw std::runtime_error(std::string(label) + ": nonzero tail metadata");
    } else if (std::find(status.begin(), status.end(), expected_error) == status.end()) {
      throw std::runtime_error(std::string(label) + ": expected error was not reported");
    }
    std::cout << label << " passed\n";
  };
  scenario("mixed varlen", base_locations, base_requests, base_positions, base_mapping, base_info, 7, 0);
  auto locations = base_locations; locations[0] = slots;
  scenario("out of bounds slot", locations, base_requests, base_positions, base_mapping, base_info, 7, 1);
  auto requests = base_requests; requests[1] = request_count;
  scenario("out of bounds request", base_locations, requests, base_positions, base_mapping, base_info, 7, 2);
  locations = base_locations; locations[0] += 1;
  scenario("slot mismatch", locations, base_requests, base_positions, base_mapping, base_info, 7, 3);
  locations = base_locations; locations[1] = locations[0];
  auto positions = base_positions; positions[1] = positions[0];
  auto info = base_info; info[1 + bucket + 1] = positions[0];
  scenario("duplicate target", locations, base_requests, positions, base_mapping, info, 7, 4);
  auto mapping = base_mapping; mapping[reqs[0] * context] = -1;
  scenario("invalid prefix", base_locations, base_requests, base_positions, mapping, base_info, 7, 6);
  mapping = base_mapping; mapping[reqs[1] * context] = base_locations[0];
  scenario("target aliases visible prefix", base_locations, base_requests, base_positions, mapping, base_info, 7, 7);
  positions = base_positions; locations = base_locations;
  for (int row = 1; row < bucket; ++row) positions[row] = locations[row] = std::numeric_limits<int32_t>::max();
  scenario("valid rows tail", locations, base_requests, positions, base_mapping, base_info, 1, 0);
  if (wide) {
    scenario("wide slot overflow", base_locations, base_requests, base_positions, base_mapping, base_info, 7, 1, 0);
    scenario("wide request overflow", base_locations, base_requests, base_positions, base_mapping, base_info, 7, 2, 1);
    scenario("wide position overflow", base_locations, base_requests, base_positions, base_mapping, base_info, 7, 1, 2);
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
    HbmArena arena(runtime.device_id(), 32ULL * 1024 * 1024);
    run(runtime, arena, false);
    run(runtime, arena, true);
    std::cout << "device metadata validation and tail gate passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
