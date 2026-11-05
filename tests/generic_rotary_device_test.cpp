// SPDX-License-Identifier: Apache-2.0
// Frozen same-input direct-FP64 BF16 reference and tail-write audit for RoPE.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <synapse_common_types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}
uint16_t bf16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
uint16_t bf16_direct(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite FP64 reference");
  const auto bits = std::bit_cast<uint64_t>(value);
  const uint16_t sign = (bits >> 48) & 0x8000;
  const uint64_t exponent = (bits >> 52) & 0x7ff;
  const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent < 897) return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(std::abs(value), 133)));
  if (exponent > 1150) return sign | 0x7f80;
  uint64_t payload = ((exponent - 896) << 7) | (fraction >> 45);
  const uint64_t tail = fraction & ((uint64_t{1} << 45) - 1);
  const uint64_t midpoint = uint64_t{1} << 44;
  payload += tail > midpoint || (tail == midpoint && (payload & 1));
  return sign | static_cast<uint16_t>(payload);
}
float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

void run(DeviceRuntime& runtime, HbmArena& arena, int q_heads, int k_heads, int head_dim,
         int rotary_dim, bool neox, int rows) {
  const int bucket = std::bit_ceil(static_cast<unsigned>(rows));
  const int q_width = q_heads * head_dim, k_width = k_heads * head_dim;
  constexpr int positions_count = 67;
  constexpr uint16_t sentinel = 0x5a5a;
  std::vector<uint16_t> query(static_cast<size_t>(bucket) * q_width);
  std::vector<uint16_t> key(static_cast<size_t>(bucket) * k_width);
  std::vector<uint16_t> cache(static_cast<size_t>(positions_count) * rotary_dim);
  std::vector<uint16_t> q_result(query.size(), sentinel), k_result(key.size(), sentinel);
  std::vector<int32_t> positions(bucket, std::numeric_limits<int32_t>::max());
  for (size_t i = 0; i < query.size(); ++i)
    query[i] = bf16(static_cast<float>(static_cast<int>((i * 13 + 29) % 1025) - 512) / 256.f);
  for (size_t i = 0; i < key.size(); ++i)
    key[i] = bf16(static_cast<float>(static_cast<int>((i * 37 + 11) % 1025) - 512) / 256.f);
  for (int pos = 0; pos < positions_count; ++pos) {
    for (int i = 0; i < rotary_dim / 2; ++i) {
      const double angle = pos * std::pow(10000.0, -2.0 * i / rotary_dim);
      cache[static_cast<size_t>(pos) * rotary_dim + i] = bf16(static_cast<float>(std::cos(angle)));
      cache[static_cast<size_t>(pos) * rotary_dim + rotary_dim / 2 + i] = bf16(static_cast<float>(std::sin(angle)));
    }
  }
  for (int row = 0; row < rows; ++row) positions[row] = (row * 17 + 3) % positions_count;
  const std::array<uint64_t, 7> sizes{query.size() * 2, key.size() * 2, positions.size() * 4,
                                      cache.size() * 2, 4, q_result.size() * 2, k_result.size() * 2};
  arena.reset();
  std::array<uint64_t, 7> addresses{};
  for (size_t i = 0; i < sizes.size(); ++i) addresses[i] = arena.allocate(sizes[i]);
  arena.copy_h2d_sync(runtime.h2d_stream(), query.data(), addresses[0], sizes[0]);
  arena.copy_h2d_sync(runtime.h2d_stream(), key.data(), addresses[1], sizes[1]);
  arena.copy_h2d_sync(runtime.h2d_stream(), positions.data(), addresses[2], sizes[2]);
  arena.copy_h2d_sync(runtime.h2d_stream(), cache.data(), addresses[3], sizes[3]);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 7> tensors{};
  std::array<synSectionHandle, 7> sections{};
  std::array<synLaunchTensorInfo, 7> launch{};
  std::array<std::string, 7> names;
  const std::string prefix = "generic_rope_q" + std::to_string(q_heads) + "_k" + std::to_string(k_heads) +
      "_h" + std::to_string(head_dim) + "_r" + std::to_string(rotary_dim) + "_s" + std::to_string(neox) +
      "_b" + std::to_string(bucket);
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = i == 2 || i == 4 ? syn_type_int32 : syn_type_bf16;
    desc.m_dims = i == 2 || i == 4 ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i == 2 ? bucket : i == 4 ? 1 :
        i == 3 ? rotary_dim : i == 0 || i == 5 ? q_width : k_width;
    if (desc.m_dims == 2) desc.m_sizes[1] = desc.m_minSizes[1] = i == 3 ? positions_count : bucket;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    for (unsigned dim = 0; dim < desc.m_dims; ++dim) launch[i].tensorSize[dim] = desc.m_sizes[dim];
  }
  struct Params { int32_t head_dim, rotary_dim, is_neox; } params{head_dim, rotary_dim, neox};
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 5, 5, 2, &params, sizeof(params),
                      "gk_rotary_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  double worst = 0, worst_floor = 0, worst_abs = 0;
  for (int valid : {rows, 1}) {
    const int32_t valid_rows = valid;
    std::fill(q_result.begin(), q_result.end(), sentinel);
    std::fill(k_result.begin(), k_result.end(), sentinel);
    arena.copy_h2d_sync(runtime.h2d_stream(), &valid_rows, addresses[4], 4);
    arena.copy_h2d_sync(runtime.h2d_stream(), q_result.data(), addresses[5], sizes[5]);
    arena.copy_h2d_sync(runtime.h2d_stream(), k_result.data(), addresses[6], sizes[6]);
    check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(), arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[5], q_result.data(), sizes[5]);
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[6], k_result.data(), sizes[6]);
    for (int which = 0; which < 2; ++which) {
      const auto& source = which ? key : query;
      const auto& actual = which ? k_result : q_result;
      const int width = which ? k_width : q_width;
      for (int row = 0; row < bucket; ++row) {
        double norm2 = 0, error2 = 0, floor2 = 0;
        for (int col = 0; col < width; ++col) {
          const size_t offset = static_cast<size_t>(row) * width + col;
          if (row >= valid) {
            if (actual[offset] != sentinel) throw std::runtime_error("RoPE tail row was modified");
            continue;
          }
          const int in_head = col % head_dim, head_base = col - in_head;
          double truth = widen(source[offset]), baseline = truth;
          if (in_head < rotary_dim) {
            const int half = rotary_dim / 2;
            const bool first = neox ? in_head < half : in_head % 2 == 0;
            const int pair = neox ? in_head % half : in_head / 2;
            const int a_index = neox ? pair : 2 * pair;
            const int b_index = neox ? half + pair : 2 * pair + 1;
            const double a = widen(source[static_cast<size_t>(row) * width + head_base + a_index]);
            const double b = widen(source[static_cast<size_t>(row) * width + head_base + b_index]);
            const double c = widen(cache[static_cast<size_t>(positions[row]) * rotary_dim + pair]);
            const double s = widen(cache[static_cast<size_t>(positions[row]) * rotary_dim + half + pair]);
            const double ac = widen(bf16_direct(a * c)), bs = widen(bf16_direct(b * s));
            const double bc = widen(bf16_direct(b * c)), as = widen(bf16_direct(a * s));
            truth = widen(bf16_direct(first ? ac - bs : bc + as));
            const float fac = widen(bf16(static_cast<float>(a) * static_cast<float>(c)));
            const float fbs = widen(bf16(static_cast<float>(b) * static_cast<float>(s)));
            const float fbc = widen(bf16(static_cast<float>(b) * static_cast<float>(c)));
            const float fas = widen(bf16(static_cast<float>(a) * static_cast<float>(s)));
            baseline = widen(bf16(first ? fac - fbs : fbc + fas));
          }
          const double value = widen(actual[offset]);
          if (!std::isfinite(value) || !std::isfinite(truth) || !std::isfinite(baseline))
            throw std::runtime_error("nonfinite rotary output or reference");
          norm2 += truth * truth;
          error2 += (value - truth) * (value - truth);
          floor2 += (baseline - truth) * (baseline - truth);
          worst_abs = std::max(worst_abs, std::abs(value - truth));
        }
        if (row < valid) {
          const double denom = std::max(std::sqrt(norm2), 1e-30);
          const double error = std::sqrt(error2) / denom, floor = std::sqrt(floor2) / denom;
          if (error > std::max(3e-3, 2 * floor) || error > 1e-2)
            throw std::runtime_error("RoPE numerical gate failed");
          worst = std::max(worst, error);
          worst_floor = std::max(worst_floor, floor);
        }
      }
    }
  }
  std::cout << "q/k " << q_heads << "/" << k_heads << " head_dim " << head_dim
            << " rotary_dim " << rotary_dim << " neox " << neox << " rows " << rows
            << " max_row_relative_l2 " << worst << " cpu_fp32_floor " << worst_floor
            << " max_abs " << worst_abs << " tail writes 0\n";
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
}
}  // namespace

int main() {
  try {
    if (bf16_direct(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -32)) != 0x3f81)
      throw std::runtime_error("FP64-to-BF16 midpoint check failed");
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 256ULL * 1024 * 1024);
    for (int rows : {1, 7, 33, 137, 1025}) run(runtime, arena, 56, 8, 128, 128, true, rows);
    for (int rows : {1, 7, 33, 137, 1025}) run(runtime, arena, 56, 8, 128, 128, false, rows);
    run(runtime, arena, 4, 2, 96, 62, true, 7);
    run(runtime, arena, 4, 2, 96, 62, false, 7);
    run(runtime, arena, 2, 1, 260, 258, true, 7);
    run(runtime, arena, 2, 1, 260, 258, false, 7);
    run(runtime, arena, 32, 8, 64, 32, true, 33);
    run(runtime, arena, 32, 8, 64, 32, false, 33);
    std::cout << "parameterized RoPE numerical and tail gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
