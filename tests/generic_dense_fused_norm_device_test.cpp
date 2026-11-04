// SPDX-License-Identifier: Apache-2.0
// Independent FP64 and same-boundary FP32 references for fused residual RMSNorm.
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

// Direct binary64-to-BF16 RNE; casting to FP32 first can round twice.
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

size_t check_pure(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows,
                  float epsilon, uint64_t residual, uint64_t gamma,
                  const std::vector<uint16_t>& expected) {
  const auto output = arena.allocate(expected.size() * sizeof(uint16_t));
  const std::array<uint64_t, 3> addresses{residual, gamma, output};
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 3> tensors{};
  std::array<synSectionHandle, 3> sections{};
  std::array<synLaunchTensorInfo, 3> launch{};
  const std::string prefix = "generic_pure_norm_w" + std::to_string(width) + "_r" + std::to_string(rows) +
                             "_e" + std::to_string(std::bit_cast<uint32_t>(epsilon));
  const std::array<std::string, 3> names{prefix + "_t0", prefix + "_t1", prefix + "_t2"};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_bf16;
    desc.m_dims = i == 1 ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = width;
    if (i != 1) desc.m_sizes[1] = desc.m_minSizes[1] = rows;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = width;
    if (i != 1) launch[i].tensorSize[1] = rows;
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, &epsilon, sizeof(epsilon),
                      "gk_rms_norm_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  std::vector<uint16_t> actual(expected.size());
  arena.copy_d2h_sync(runtime.d2h_stream(), output, actual.data(), actual.size() * sizeof(uint16_t));
  size_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) mismatches += actual[i] != expected[i];
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  return mismatches;
}

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t rows, float epsilon) {
  const size_t count = static_cast<size_t>(width) * rows;
  std::vector<uint16_t> x(count), residual(count), gamma(width), actual_residual(count), actual_norm(count);
  for (size_t i = 0; i < count; ++i) {
    const int32_t a = static_cast<int32_t>((i * 48271ULL + 17) % 200001) - 100000;
    const int32_t b = static_cast<int32_t>((i * 69621ULL + 37) % 200001) - 100000;
    x[i] = bf16(std::ldexp(static_cast<float>(a), static_cast<int>((i / 127) % 7) - 18));
    residual[i] = bf16(std::ldexp(static_cast<float>(b), static_cast<int>((i / 131) % 7) - 18));
  }
  for (uint32_t i = 0; i < width; ++i) gamma[i] = bf16(0.5f + static_cast<float>((i * 31) % 1021) / 1024.f);
  arena.reset();
  const std::array<uint64_t, 5> addresses{
      arena.allocate(count * 2), arena.allocate(count * 2), arena.allocate(width * 2),
      arena.allocate(count * 2), arena.allocate(count * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), x.data(), addresses[0], count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), residual.data(), addresses[1], count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), gamma.data(), addresses[2], width * 2);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 5> tensors{};
  std::array<synSectionHandle, 5> sections{};
  std::array<synLaunchTensorInfo, 5> launch{};
  const std::string prefix = "generic_norm_w" + std::to_string(width) + "_r" + std::to_string(rows) +
                             "_e" + std::to_string(std::bit_cast<uint32_t>(epsilon));
  const std::array<std::string, 5> names{prefix + "_t0", prefix + "_t1", prefix + "_t2",
                                         prefix + "_t3", prefix + "_t4"};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_bf16;
    desc.m_dims = i == 2 ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = width;
    if (i != 2) desc.m_sizes[1] = desc.m_minSizes[1] = rows;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = width;
    if (i != 2) launch[i].tensorSize[1] = rows;
  }
  const auto node = prefix + "_n0";
  check(synNodeCreate(graph, tensors.data(), &tensors[3], 3, 2, &epsilon, sizeof(epsilon),
                      "gk_fused_add_rms_norm_bf16_v1", node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), addresses[3], actual_residual.data(), count * 2);
  arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_norm.data(), count * 2);
  size_t residual_mismatches = 0;
  size_t unrounded_reference_differences = 0;
  double max_error = 0, max_floor = 0, max_absolute = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    double sum64 = 0;
    float sum32 = 0;
    for (uint32_t col = 0; col < width; ++col) {
      const size_t i = static_cast<size_t>(row) * width + col;
      const auto expected_residual = bf16(widen(x[i]) + widen(residual[i]));
      residual_mismatches += actual_residual[i] != expected_residual;
      const float value = widen(expected_residual);
      sum64 += static_cast<double>(value) * value;
      sum32 += value * value;
    }
    const double inv64 = 1.0 / std::sqrt(sum64 / width + static_cast<double>(epsilon));
    const float inv32 = 1.f / std::sqrt(sum32 / static_cast<float>(width) + epsilon);
    double unrounded_inverse = 0;
    if (row == 0 && width > 1 && rows > 1) {
      double unrounded_sum = 0;
      for (uint32_t col = 0; col < width; ++col) {
        const double value = static_cast<double>(widen(x[col]) + widen(residual[col]));
        unrounded_sum += value * value;
      }
      unrounded_inverse = 1.0 / std::sqrt(unrounded_sum / width + static_cast<double>(epsilon));
    }
    double norm2 = 0, err2 = 0, floor2 = 0;
    for (uint32_t col = 0; col < width; ++col) {
      const size_t i = static_cast<size_t>(row) * width + col;
      const float value = widen(bf16(widen(x[i]) + widen(residual[i])));
      const double truth = widen(bf16_direct(static_cast<double>(value) * inv64 * widen(gamma[col])));
      if (row == 0 && width > 1 && rows > 1) {
        const double unrounded = static_cast<double>(widen(x[i]) + widen(residual[i]));
        unrounded_reference_differences +=
            bf16_direct(unrounded * unrounded_inverse * widen(gamma[col])) != bf16_direct(truth);
      }
      const double reference32 = widen(bf16(value * inv32 * widen(gamma[col])));
      const double result = widen(actual_norm[i]);
      if (!std::isfinite(result) || !std::isfinite(truth) || !std::isfinite(reference32))
        throw std::runtime_error("nonfinite norm result");
      norm2 += truth * truth;
      err2 += (result - truth) * (result - truth);
      floor2 += (reference32 - truth) * (reference32 - truth);
      max_absolute = std::max(max_absolute, std::abs(result - truth));
    }
    const double denom = std::max(std::sqrt(norm2), 1e-30);
    max_error = std::max(max_error, std::sqrt(err2) / denom);
    max_floor = std::max(max_floor, std::sqrt(floor2) / denom);
  }
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  const size_t pure_mismatches = check_pure(runtime, arena, width, rows, epsilon,
                                            addresses[3], addresses[2], actual_norm);
  std::cout << "width " << width << " rows " << rows << " residual_bit_differences " << residual_mismatches
            << " pure_bit_differences " << pure_mismatches << " max_row_relative_l2 " << max_error
            << " cpu_fp32_floor " << max_floor << " max_abs " << max_absolute
            << " unrounded_reference_differences " << unrounded_reference_differences << '\n';
  if (residual_mismatches || pure_mismatches || (width > 1 && rows > 1 && !unrounded_reference_differences) ||
      max_error > std::max(3e-3, 2 * max_floor) || max_error > 1e-2)
    throw std::runtime_error("fused residual RMSNorm failed the numerical gate");
}
}  // namespace

int main() {
  try {
    if (bf16_direct(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -32)) != 0x3f81)
      throw std::runtime_error("FP64-to-BF16 midpoint check failed");
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 192ULL * 1024 * 1024);
    for (const auto [width, rows] : {std::pair{1, 1}, {127, 7}, {129, 7}, {8192, 7},
                                    {7168, 1}, {7168, 7}, {7168, 33}, {7168, 137}, {7168, 1025}})
      run(runtime, arena, width, rows, 1e-6f);
    run(runtime, arena, 7168, 17, 1e-5f);
    run(runtime, arena, 129, 3, 0.5f);
    std::cout << "fused residual RMSNorm numerical gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
