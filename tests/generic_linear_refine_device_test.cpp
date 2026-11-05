// SPDX-License-Identifier: Apache-2.0
// Exercise alternate MME reductions and device refinement on frozen BF16 inputs.
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
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}
uint16_t narrow(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
uint16_t direct(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
  const auto bits = std::bit_cast<uint64_t>(value);
  const uint16_t sign = (bits >> 48) & 0x8000;
  const uint64_t exponent = (bits >> 52) & 0x7ff;
  const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent < 897) return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(std::abs(value), 133)));
  if (exponent > 1150) return sign | 0x7f80;
  uint64_t payload = ((exponent - 896) << 7) | (fraction >> 45);
  const uint64_t tail = fraction & ((uint64_t{1} << 45) - 1), halfway = uint64_t{1} << 44;
  payload += tail > halfway || (tail == halfway && (payload & 1));
  return sign | static_cast<uint16_t>(payload);
}
float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t columns) {
  uint32_t seed = 0x1729a5u;
  auto sample = [&] {
    seed = 1664525u * seed + 1013904223u;
    return (static_cast<int32_t>((seed >> 8) % 20001) - 10000) / 8192.f;
  };
  std::vector<uint16_t> input(width), weight(static_cast<size_t>(width) * columns);
  for (auto& value : input) value = narrow(sample());
  for (auto& value : weight) value = narrow(sample());
  std::array<std::vector<uint16_t>, 3> outputs;
  for (auto& output : outputs) output.resize(columns);
  const std::array<uint64_t, 5> addresses{arena.allocate(width * 2), arena.allocate(weight.size() * 2),
                                          arena.allocate(columns * 2), arena.allocate(columns * 2),
                                          arena.allocate(columns * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), input.data(), addresses[0], width * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), weight.data(), addresses[1], weight.size() * 2);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 5> tensors{};
  std::array<synSectionHandle, 5> sections{};
  std::array<synLaunchTensorInfo, 5> launch{};
  std::array<std::string, 5> names;
  std::vector<synTensor> internal;
  std::vector<std::string> internal_names;
  internal.reserve(16);
  internal_names.reserve(16);
  const auto prefix = "linear_refine_k" + std::to_string(width) + "_n" + std::to_string(columns);
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = syn_type_bf16;
    desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i < 2 ? width : columns;
    desc.m_sizes[1] = desc.m_minSizes[1] = i == 1 ? columns : 1;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = desc.m_sizes[0];
    launch[i].tensorSize[1] = desc.m_sizes[1];
  }
  auto make = [&](synDataType type, uint32_t first, uint32_t second) {
    internal_names.push_back(prefix + "_t" + std::to_string(tensors.size() + internal.size()));
    synTensorDescriptor desc{};
    desc.m_name = internal_names.back().c_str();
    desc.m_dataType = type;
    desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = first;
    desc.m_sizes[1] = desc.m_minSizes[1] = second;
    synTensor result = nullptr;
    check(synTensorCreate(&result, &desc, nullptr, 0));
    internal.push_back(result);
    return result;
  };
  unsigned node_index = 0;
  auto append = [&](std::vector<synTensor> in, std::vector<synTensor> out,
                    const void* params, size_t bytes, const char* guid) {
    const auto name = prefix + "_n" + std::to_string(node_index++);
    check(synNodeCreate(graph, in.data(), out.data(), in.size(), out.size(),
                        params, bytes, guid, name.c_str(), nullptr, nullptr));
  };
  auto full = make(syn_type_float, columns, 1);
  auto first = make(syn_type_bf16, width / 2, 1), second = make(syn_type_bf16, width / 2, 1);
  auto first_weight = make(syn_type_bf16, width / 2, columns);
  auto second_weight = make(syn_type_bf16, width / 2, columns);
  auto partial0 = make(syn_type_float, columns, 1), partial1 = make(syn_type_float, columns, 1);
  auto sum = make(syn_type_float, columns, 1);
  synGEMMParams gemm{false, true};
  synSplitParams split{0};
  append({tensors[0], tensors[1]}, {full}, &gemm, sizeof(gemm), "gemm");
  append({tensors[0]}, {first, second}, &split, sizeof(split), "split");
  append({tensors[1]}, {first_weight, second_weight}, &split, sizeof(split), "split");
  append({first, first_weight}, {partial0}, &gemm, sizeof(gemm), "gemm");
  append({second, second_weight}, {partial1}, &gemm, sizeof(gemm), "gemm");
  append({partial0, partial1}, {sum}, nullptr, 0, "add_fwd_f32");
  append({full}, {tensors[2]}, nullptr, 0, "gk_cast_f32_to_bf16_v1");
  append({sum}, {tensors[3]}, nullptr, 0, "gk_cast_f32_to_bf16_v1");
  append({tensors[0], tensors[1], tensors[2], tensors[3], full, sum}, {tensors[4]}, nullptr, 0,
         "gk_linear_refine_bf16_v2");
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(), arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  for (size_t i = 0; i < outputs.size(); ++i)
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[i + 2], outputs[i].data(), columns * 2);
  size_t disagreements = 0, refined_wrong = 0, initial_wrong = 0, agreement_changes = 0;
  std::vector<uint16_t> references(columns);
  double error = 0, norm = 0, maximum = 0;
  for (uint32_t col = 0; col < columns; ++col) {
    double sum64 = 0;
    for (uint32_t k = 0; k < width; ++k)
      sum64 += static_cast<double>(widen(input[k])) * widen(weight[static_cast<size_t>(col) * width + k]);
    const auto reference = direct(sum64);
    references[col] = reference;
    const double actual = widen(outputs[2][col]), expected = widen(reference);
    if (!std::isfinite(actual) || !std::isfinite(expected)) throw std::runtime_error("nonfinite output");
    initial_wrong += outputs[0][col] != reference;
    if (outputs[0][col] != outputs[1][col]) {
      ++disagreements;
      refined_wrong += outputs[2][col] != reference;
    } else if (outputs[2][col] != outputs[0][col]) {
      ++agreement_changes;
      refined_wrong += outputs[2][col] != reference;
    }
    const auto delta = actual - expected;
    error += delta * delta;
    norm += expected * expected;
    maximum = std::max(maximum, std::abs(delta));
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-60));
  std::cout << "linear refine width " << width << " columns " << columns << " disagreements " << disagreements
            << " original_mismatches " << initial_wrong << " agreement_changes " << agreement_changes
            << " refined_wrong " << refined_wrong
            << " relative_l2 " << relative << " max_abs " << maximum << std::endl;
  if (relative > 3e-3 || refined_wrong) throw std::runtime_error("linear refinement accuracy failed");
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : internal) check(synTensorDestroy(tensor));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));

  // Force a small set of disagreements so the compensated branch is tested
  // even when the two MME trees happen to round alike on this random input.
  const uint32_t forced = std::min<uint32_t>(columns, 32);
  auto alternatives = outputs[0];
  for (uint32_t col = 0; col < forced; ++col) alternatives[col] ^= 1;
  arena.copy_h2d_sync(runtime.h2d_stream(), outputs[0].data(), addresses[2], columns * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), alternatives.data(), addresses[3], columns * 2);
  graph = nullptr;
  recipe = nullptr;
  std::array<synTensor, 7> forced_tensors{};
  std::array<synSectionHandle, 7> forced_sections{};
  std::array<synLaunchTensorInfo, 7> forced_launch{};
  std::array<std::string, 7> forced_names;
  const std::array<uint64_t, 7> forced_addresses{
      addresses[0], addresses[1], addresses[2], addresses[3], arena.allocate(columns * 4),
      arena.allocate(columns * 4), addresses[4]};
  const std::vector<float> far_from_midpoint(columns, 1.f);
  arena.copy_h2d_sync(runtime.h2d_stream(), far_from_midpoint.data(), forced_addresses[4], columns * 4);
  arena.copy_h2d_sync(runtime.h2d_stream(), far_from_midpoint.data(), forced_addresses[5], columns * 4);
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < forced_tensors.size(); ++i) {
    forced_names[i] = prefix + "_forced_t" + std::to_string(i);
    check(synSectionCreate(&forced_sections[i], 0, graph));
    check(synSectionSetPersistent(forced_sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = forced_names[i].c_str();
    desc.m_dataType = i == 4 || i == 5 ? syn_type_float : syn_type_bf16;
    desc.m_dims = 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i < 2 ? width : columns;
    desc.m_sizes[1] = desc.m_minSizes[1] = i == 1 ? columns : 1;
    check(synTensorCreate(&forced_tensors[i], &desc, forced_sections[i], 0));
    forced_launch[i].tensorName = forced_names[i].c_str();
    forced_launch[i].tensorType = DATA_TENSOR;
    forced_launch[i].pTensorAddress = forced_addresses[i];
    forced_launch[i].tensorSize[0] = desc.m_sizes[0];
    forced_launch[i].tensorSize[1] = desc.m_sizes[1];
  }
  const auto forced_node = prefix + "_forced_n0";
  check(synNodeCreate(graph, forced_tensors.data(), &forced_tensors[6], 6, 1, nullptr, 0,
                      "gk_linear_refine_bf16_v2", forced_node.c_str(), nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, (prefix + "_forced").c_str(), nullptr));
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, forced_launch.data(), forced_launch.size());
  check(synLaunch(runtime.compute_stream(), forced_launch.data(), forced_launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], outputs[2].data(), columns * 2);
  for (uint32_t col = 0; col < columns; ++col) {
    const auto expected = col < forced ? references[col] : outputs[0][col];
    if (outputs[2][col] != expected) throw std::runtime_error("forced refinement disagrees with reference");
  }
  std::cout << "forced compensated rows " << forced << " passed" << std::endl;
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : forced_tensors) check(synTensorDestroy(tensor));
  for (auto section : forced_sections) check(synSectionDestroy(section));
}
}  // namespace

int main() {
  try {
    if (direct(1.0 + std::ldexp(1.0, -8) + std::ldexp(1.0, -32)) != 0x3f81)
      throw std::runtime_error("FP64-to-BF16 midpoint check failed");
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 256ULL * 1024 * 1024);
    run(runtime, arena, 1024, 512);
    run(runtime, arena, 7168, 1024);
    std::cout << "linear refinement frozen-input device gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
