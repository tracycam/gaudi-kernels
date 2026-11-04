// SPDX-License-Identifier: Apache-2.0
// Independent direct-FP64 BF16 reference for parameterized SiLU and multiply.
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
  const uint64_t half = uint64_t{1} << 44;
  payload += tail > half || (tail == half && (payload & 1));
  return sign | static_cast<uint16_t>(payload);
}
float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows) {
  const size_t count = static_cast<size_t>(width) * rows;
  std::vector<uint16_t> input(count * 2), actual(count);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      const size_t i = static_cast<size_t>(row) * width + col;
      float gate = static_cast<float>(static_cast<int32_t>((i * 48271ULL + 17) % 2049) - 1024) / 64.f;
      if (col % 131 == 0) gate = 0.69140625f;
      if (col % 131 == 1) gate = 0.6953125f;
      if (col % 131 == 2) gate = -100.f;
      if (col % 131 == 3) gate = 100.f;
      const float up = static_cast<float>(static_cast<int32_t>((i * 69621ULL + 37) % 1025) - 512) / 256.f;
      input[static_cast<size_t>(row) * width * 2 + col] = bf16(gate);
      input[static_cast<size_t>(row) * width * 2 + width + col] = bf16(up);
    }
  }
  arena.reset();
  const std::array<uint64_t, 2> addresses{arena.allocate(count * 4), arena.allocate(count * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), input.data(), addresses[0], count * 4);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 2> tensors{};
  std::array<synSectionHandle, 2> sections{};
  std::array<synLaunchTensorInfo, 2> launch{};
  const std::string prefix = "generic_silu_w" + std::to_string(width) + "_r" + std::to_string(rows);
  const std::array<std::string, 2> names{prefix + "_t0", prefix + "_t1"};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_bf16;
    desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 ? width * 2 : width;
    desc.m_sizes[1] = desc.m_minSizes[1] = rows;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = desc.m_sizes[0];
    launch[i].tensorSize[1] = rows;
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, &tensors[0], &tensors[1], 1, 1, nullptr, 0,
                      "gk_silu_and_mul_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), addresses[1], actual.data(), count * 2);
  double max_error = 0, max_floor = 0, max_absolute = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    double norm2 = 0, err2 = 0, floor2 = 0;
    for (uint32_t col = 0; col < width; ++col) {
      const size_t i = static_cast<size_t>(row) * width + col;
      const float gate = widen(input[static_cast<size_t>(row) * width * 2 + col]);
      const float up = widen(input[static_cast<size_t>(row) * width * 2 + width + col]);
      const double wide_gate = gate;
      const float silu32 = gate / (1.f + std::exp(-gate));
      const double silu64 = wide_gate / (1.0 + std::exp(-wide_gate));
      const double truth = widen(bf16_direct(static_cast<double>(widen(bf16_direct(silu64))) * up));
      const double reference32 = widen(bf16(widen(bf16(silu32)) * up));
      const double result = widen(actual[i]);
      if (!std::isfinite(truth) || !std::isfinite(reference32) || !std::isfinite(result))
        throw std::runtime_error("nonfinite SiLU result");
      norm2 += truth * truth;
      err2 += (result - truth) * (result - truth);
      floor2 += (reference32 - truth) * (reference32 - truth);
      max_absolute = std::max(max_absolute, std::abs(result - truth));
    }
    const double denom = std::max(std::sqrt(norm2), 1e-30);
    max_error = std::max(max_error, std::sqrt(err2) / denom);
    max_floor = std::max(max_floor, std::sqrt(floor2) / denom);
  }
  std::cout << "width " << width << " rows " << rows << " max_row_relative_l2 " << max_error
            << " cpu_fp32_floor " << max_floor << " max_abs " << max_absolute << '\n';
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  if (max_error > std::max(3e-3, 2 * max_floor) || max_error > 1e-2)
    throw std::runtime_error("SiLU and multiply failed the numerical gate");
}
}  // namespace

int main() {
  try {
    if (bf16_direct(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -32)) != 0x3f81)
      throw std::runtime_error("FP64-to-BF16 midpoint check failed");
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 256ULL * 1024 * 1024);
    for (const auto [width, rows] : {std::pair{1, 1}, {7, 3}, {127, 7}, {128, 1}, {129, 7},
                                    {22016, 1}, {22016, 7}, {22016, 33}, {22016, 137}, {22016, 1025}})
      run(runtime, arena, width, rows);
    std::cout << "SiLU and multiply numerical gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
