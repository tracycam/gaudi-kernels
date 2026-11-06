// SPDX-License-Identifier: Apache-2.0
// Frozen same-input FP64/FP32 test for online causal GQA over scattered KV slots.
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
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
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

std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t>, std::vector<uint16_t>> replay_rows;

void write_and_audit(DeviceRuntime& runtime, HbmArena& arena, uint32_t width, uint32_t bucket,
                     uint32_t rows, const std::vector<int32_t>& mapping, uint64_t key_address,
                     uint64_t value_address, uint64_t metadata_address,
                     std::vector<uint16_t>& key, std::vector<uint16_t>& value) {
  if (rows > mapping.size()) throw std::runtime_error("chain test needs unique locations");
  const uint32_t slots = key.size() / width;
  const size_t count = static_cast<size_t>(width) * bucket;
  std::vector<uint16_t> new_key(count), new_value(count), actual_key(key.size()), actual_value(value.size());
  std::vector<int32_t> locations(bucket, 0), source_for_slot(slots, -1);
  for (uint32_t row = 0; row < rows; ++row) {
    const auto slot = mapping[row];
    if (source_for_slot[slot] >= 0) throw std::runtime_error("chain test location is duplicated");
    locations[row] = slot;
    source_for_slot[slot] = row;
    for (uint32_t col = 0; col < width; ++col) {
      const auto src = static_cast<size_t>(row) * width + col;
      const auto dst = static_cast<size_t>(slot) * width + col;
      new_key[src] = bf16(static_cast<float>(static_cast<int32_t>((src * 13 + 47) % 257) - 128) / 64.f);
      new_value[src] = bf16(static_cast<float>(static_cast<int32_t>((src * 23 + 71) % 257) - 128) / 64.f);
      key[dst] = new_key[src];
      value[dst] = new_value[src];
    }
  }
  const std::array<uint64_t, 6> addresses{
      arena.allocate(count * 2), arena.allocate(count * 2), arena.allocate(bucket * 4),
      metadata_address, key_address, value_address};
  arena.copy_h2d_sync(runtime.h2d_stream(), new_key.data(), addresses[0], count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), new_value.data(), addresses[1], count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), locations.data(), addresses[2], bucket * 4);
  const int32_t valid = rows;
  arena.copy_h2d_sync(runtime.h2d_stream(), &valid, metadata_address, sizeof(valid));
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 6> tensors{};
  std::array<synSectionHandle, 6> sections{};
  std::array<synLaunchTensorInfo, 6> launch{};
  const std::string prefix = "generic_gqa_chain_store_w" + std::to_string(width) + "_b" + std::to_string(bucket);
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
  check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), key_address, actual_key.data(), key.size() * 2);
  arena.copy_d2h_sync(runtime.d2h_stream(), value_address, actual_value.data(), value.size() * 2);
  size_t incorrect_target = 0, touched_other = 0;
  for (uint32_t slot = 0; slot < slots; ++slot) {
    for (uint32_t col = 0; col < width; ++col) {
      const auto i = static_cast<size_t>(slot) * width + col;
      const bool bad = actual_key[i] != key[i] || actual_value[i] != value[i];
      if (source_for_slot[slot] >= 0) incorrect_target += bad;
      else touched_other += bad;
    }
  }
  std::cout << "chain valid_rows " << rows << " incorrect_target_elements " << incorrect_target
            << " touched_other_elements " << touched_other << std::endl;
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : tensors) check(synTensorDestroy(tensor));
  for (auto section : sections) check(synSectionDestroy(section));
  if (incorrect_target || touched_other) throw std::runtime_error("chain KV write audit failed");
}

void run(DeviceRuntime& runtime, HbmArena& arena, uint32_t q_heads, uint32_t kv_heads,
         uint32_t head_dim, uint32_t rows, uint32_t context, bool write_first = false,
         uint32_t position_limit = UINT32_MAX) {
  constexpr uint32_t requests = 4;
  constexpr uint16_t sentinel = 0x3e80;
  const uint32_t bucket = std::bit_ceil(rows);
  const uint32_t slots_base = std::bit_ceil(requests * context);
  const uint32_t slots = slots_base + 1;
  const uint32_t q_width = q_heads * head_dim, kv_width = kv_heads * head_dim;
  const size_t query_count = static_cast<size_t>(q_width) * bucket;
  const size_t pool_count = static_cast<size_t>(kv_width) * slots;
  std::vector<uint16_t> query(query_count), key(pool_count), value(pool_count),
      init_output(query_count, sentinel), actual(query_count), first_row(q_width);
  std::vector<int32_t> request_ids(bucket), positions(bucket), mapping(requests * context);
  for (size_t i = 0; i < query_count; ++i)
    query[i] = bf16(static_cast<float>(static_cast<int32_t>((i * 197ULL + 31) % 513) - 256) / 128.f);
  for (size_t i = 0; i < pool_count; ++i) {
    key[i] = bf16(static_cast<float>(static_cast<int32_t>((i * 97ULL + 11) % 513) - 256) / 128.f);
    value[i] = bf16(static_cast<float>(static_cast<int32_t>((i * 149ULL + 19) % 513) - 256) / 128.f);
  }
  for (uint32_t row = 0; row < rows; ++row) {
    request_ids[row] = (row * 3) % requests;
    positions[row] = row == 0 ? std::min(context - 1, position_limit) : (row * 17 + 5) % context;
  }
  if (rows > 1) {
    std::copy_n(query.begin(), q_width, query.begin() + static_cast<size_t>(rows - 1) * q_width);
    request_ids[rows - 1] = request_ids[0];
    positions[rows - 1] = positions[0];
  }
  for (uint32_t request = 0; request < requests; ++request)
    for (uint32_t token = 0; token < context; ++token)
      mapping[request * context + token] = 1 + ((request * context + token) * 199) % slots_base;
  arena.reset();
  const std::array<uint64_t, 8> addresses{
      arena.allocate(query_count * 2), arena.allocate(pool_count * 2), arena.allocate(pool_count * 2),
      arena.allocate(bucket * 4), arena.allocate(bucket * 4), arena.allocate(mapping.size() * 4),
      arena.allocate(sizeof(int32_t)), arena.allocate(query_count * 2)};
  arena.copy_h2d_sync(runtime.h2d_stream(), query.data(), addresses[0], query_count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), key.data(), addresses[1], pool_count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), value.data(), addresses[2], pool_count * 2);
  arena.copy_h2d_sync(runtime.h2d_stream(), request_ids.data(), addresses[3], bucket * 4);
  arena.copy_h2d_sync(runtime.h2d_stream(), positions.data(), addresses[4], bucket * 4);
  arena.copy_h2d_sync(runtime.h2d_stream(), mapping.data(), addresses[5], mapping.size() * 4);
  if (write_first)
    write_and_audit(runtime, arena, kv_width, bucket, rows, mapping, addresses[1], addresses[2], addresses[6], key, value);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synTensor, 8> tensors{};
  std::array<synSectionHandle, 8> sections{};
  std::array<synLaunchTensorInfo, 8> launch{};
  const std::string prefix = "generic_gqa_q" + std::to_string(q_heads) + "_d" + std::to_string(head_dim) +
                             "_b" + std::to_string(bucket) + "_c" + std::to_string(context) +
                             "_kv" + std::to_string(kv_heads) + "_mme" +
                             (write_first ? "_chain" : "");
  std::array<std::string, 8> names;
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    names[i] = prefix + "_t" + std::to_string(i);
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i].c_str();
    desc.m_dataType = i >= 3 && i <= 6 ? syn_type_int32 : syn_type_bf16;
    desc.m_dims = (i == 3 || i == 4 || i == 6) ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 || i == 7 ? q_width :
                                               i == 1 || i == 2 ? kv_width :
                                               i == 5 ? context : i == 6 ? 1 : bucket;
    if (desc.m_dims == 2)
      desc.m_sizes[1] = desc.m_minSizes[1] = i == 0 || i == 7 ? bucket : i == 5 ? requests : slots;
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    launch[i].tensorName = names[i].c_str();
    launch[i].tensorType = DATA_TENSOR;
    launch[i].pTensorAddress = addresses[i];
    launch[i].tensorSize[0] = desc.m_sizes[0];
    if (desc.m_dims == 2) launch[i].tensorSize[1] = desc.m_sizes[1];
  }
  struct Params { int32_t head_dim; float scale; } params{static_cast<int32_t>(head_dim), 1.f / std::sqrt(float(head_dim))};
  std::vector<synTensor> intermediates;
  std::vector<std::string> temporary_names;
  temporary_names.reserve(16);
  {
    auto intermediate = [&](synDataType type, std::initializer_list<uint64_t> dims) {
      temporary_names.push_back(prefix + "_i" + std::to_string(temporary_names.size()));
      synTensorDescriptor desc{};
      desc.m_name = temporary_names.back().c_str();
      desc.m_dataType = type;
      desc.m_dims = dims.size();
      size_t i = 0;
      for (auto dim : dims) { desc.m_sizes[i] = desc.m_minSizes[i] = dim; ++i; }
      synTensor result = nullptr;
      check(synTensorCreate(&result, &desc, nullptr, 0));
      intermediates.push_back(result);
      return result;
    };
    unsigned next_node = 0;
    auto append = [&](std::vector<synTensor> in, std::vector<synTensor> out,
                      const void* argument, size_t size, const char* guid) {
      const auto name = prefix + "_n" + std::to_string(next_node++);
      check(synNodeCreate(graph, in.data(), out.data(), in.size(), out.size(),
                          argument, size, guid, name.c_str(), nullptr, nullptr));
    };
    const uint64_t ratio = q_heads / kv_heads, groups = static_cast<uint64_t>(kv_heads) * bucket;
    auto key4 = intermediate(syn_type_bf16, {head_dim, context, kv_heads, bucket});
    auto value4 = intermediate(syn_type_bf16, {head_dim, context, kv_heads, bucket});
    auto key3 = intermediate(syn_type_bf16, {head_dim, context, groups});
    auto value3 = intermediate(syn_type_bf16, {head_dim, context, groups});
    auto query3 = intermediate(syn_type_bf16, {head_dim, ratio, groups});
    auto scores = intermediate(syn_type_float, {context, ratio, groups});
    auto probability = intermediate(syn_type_bf16, {context, ratio, groups});
    auto main_context = intermediate(syn_type_float, {head_dim, ratio, groups});
    auto flat_context = intermediate(syn_type_float, {q_width, bucket});
    const int32_t dimension = head_dim;
    append({tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6]},
           {key4, value4}, &dimension, sizeof(dimension), "gk_paged_gqa_mme_gather_bf16_v1");
    append({key4}, {key3}, nullptr, 0, "reshape");
    append({value4}, {value3}, nullptr, 0, "reshape");
    append({tensors[0]}, {query3}, nullptr, 0, "reshape");
    const synGEMMParams qk{false, true};
    append({query3, key3}, {scores}, &qk, sizeof(qk), "batch_gemm");
    struct SoftmaxParams { int32_t kv_heads; float scale; } softmax{static_cast<int32_t>(kv_heads), params.scale};
    append({scores, tensors[4], tensors[6]}, {probability},
           &softmax, sizeof(softmax), "gk_paged_gqa_mme_softmax_bf16_v1");
    const synGEMMParams pv{false, false};
    append({probability, value3}, {main_context}, &pv, sizeof(pv), "batch_gemm");
    append({main_context}, {flat_context}, nullptr, 0, "reshape");
    append({flat_context}, {tensors[7]}, nullptr, 0, "gk_cast_f32_to_bf16_v1");
  }
  check(synGraphCompile(&recipe, graph, prefix.c_str(), nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  for (const uint32_t valid : {rows, uint32_t{1}}) {
    const int32_t metadata = valid;
    arena.copy_h2d_sync(runtime.h2d_stream(), &metadata, addresses[6], sizeof(metadata));
    arena.copy_h2d_sync(runtime.h2d_stream(), init_output.data(), addresses[7], query_count * 2);
    check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                    arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[7], actual.data(), query_count * 2);
    if (valid == rows) {
      std::copy_n(actual.begin(), q_width, first_row.begin());
      const auto replay_key = std::make_tuple(q_heads, kv_heads, head_dim, context, write_first, static_cast<uint32_t>(positions[0]));
      const auto [it, inserted] = replay_rows.emplace(replay_key, first_row);
      if (!inserted && it->second != first_row) {
        double norm = 0, difference = 0, maximum = 0;
        size_t changed = 0;
        for (size_t i = 0; i < first_row.size(); ++i) {
          const double actual = widen(first_row[i]), reference = widen(it->second[i]);
          if (!std::isfinite(actual) || !std::isfinite(reference))
            throw std::runtime_error("nonfinite same-prefix attention row");
          norm += reference * reference;
          difference += (actual - reference) * (actual - reference);
          maximum = std::max(maximum, std::abs(actual - reference));
          changed += first_row[i] != it->second[i];
        }
        const double relative = std::sqrt(difference / std::max(norm, 1e-60));
        std::cout << "MME same-prefix q_heads " << q_heads << " context " << context
                  << " changed " << changed << " relative_l2 " << relative
                  << " max_abs " << maximum << std::endl;
        if (relative > 3e-3) throw std::runtime_error("MME same-prefix row exceeds numerical gate");
      }
      if (rows > 1 && !std::equal(first_row.begin(), first_row.end(),
                                  actual.begin() + static_cast<size_t>(rows - 1) * q_width))
        throw std::runtime_error("last valid row differs from M=1 for the same prefix");
    }
    else if (!std::equal(first_row.begin(), first_row.end(), actual.begin()))
      throw std::runtime_error("first row changed when valid_rows changed");
    size_t tail_changes = 0;
    for (size_t i = static_cast<size_t>(valid) * q_width; i < query_count; ++i)
      tail_changes += actual[i] != sentinel;
    double max_error = 0, max_floor = 0, max_absolute = 0;
    std::vector<double> scores64(context);
    std::vector<float> scores32(context);
    for (uint32_t row = 0; row < valid; ++row) {
      double row_norm = 0, row_error = 0, row_floor = 0;
      const int request = request_ids[row], length = positions[row] + 1;
      for (uint32_t head = 0; head < q_heads; ++head) {
        const uint32_t kv_head = head / (q_heads / kv_heads);
        double max64 = -INFINITY;
        float max32 = -INFINITY;
        for (int token = 0; token < length; ++token) {
          const int slot = mapping[request * context + token];
          double sum64 = 0;
          float sum32 = 0;
          for (uint32_t col = 0; col < head_dim; ++col) {
            const float q = widen(query[static_cast<size_t>(row) * q_width + head * head_dim + col]);
            const float k = widen(key[static_cast<size_t>(slot) * kv_width + kv_head * head_dim + col]);
            sum64 += static_cast<double>(q) * k;
            sum32 += q * k;
          }
          scores64[token] = sum64 * params.scale;
          scores32[token] = sum32 * params.scale;
          max64 = std::max(max64, scores64[token]);
          max32 = std::max(max32, scores32[token]);
        }
        std::array<double, 128> acc64{};
        std::array<float, 128> acc32{};
        double denominator64 = 0;
        float denominator32 = 0;
        for (int token = 0; token < length; ++token)
          denominator32 += std::exp(scores32[token] - max32);
        for (int token = 0; token < length; ++token) {
          const int slot = mapping[request * context + token];
          const double weight64 = std::exp(scores64[token] - max64);
          const float weight32 = widen(bf16(std::exp(scores32[token] - max32) / denominator32));
          denominator64 += weight64;
          for (uint32_t col = 0; col < head_dim; ++col) {
            const float v = widen(value[static_cast<size_t>(slot) * kv_width + kv_head * head_dim + col]);
            acc64[col] += weight64 * v;
            acc32[col] += weight32 * v;
          }
        }
        for (uint32_t col = 0; col < head_dim; ++col) {
          const double truth = widen(bf16_direct(acc64[col] / denominator64));
          const double reference32 = widen(bf16(acc32[col]));
          const double result = widen(actual[static_cast<size_t>(row) * q_width + head * head_dim + col]);
          if (!std::isfinite(result) || !std::isfinite(truth) || !std::isfinite(reference32))
            throw std::runtime_error("nonfinite attention result");
          row_norm += truth * truth;
          row_error += (result - truth) * (result - truth);
          row_floor += (reference32 - truth) * (reference32 - truth);
          max_absolute = std::max(max_absolute, std::abs(result - truth));
        }
      }
      const double denom = std::max(std::sqrt(row_norm), 1e-30);
      max_error = std::max(max_error, std::sqrt(row_error) / denom);
      max_floor = std::max(max_floor, std::sqrt(row_floor) / denom);
    }
    std::cout << "MME q_heads " << q_heads << " kv_heads " << kv_heads << " head_dim " << head_dim
              << " bucket " << bucket << " valid_rows " << valid << " max_context " << context
              << " max_row_relative_l2 " << max_error << " cpu_fp32_floor " << max_floor
              << " max_abs " << max_absolute << " tail_changed " << tail_changes << std::endl;
    if (tail_changes || max_error > std::max(3e-3, 2 * max_floor) || max_error > 1e-2)
      throw std::runtime_error("paged GQA numerical or tail gate failed");
  }
  check(synRecipeDestroy(recipe));
  check(synGraphDestroy(graph));
  for (auto tensor : intermediates) check(synTensorDestroy(tensor));
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
    for (const auto context : {1u, 37u, 63u, 64u, 65u, 127u, 128u, 129u, 137u, 1053u, 4096u})
      run(runtime, arena, 56, 8, 128, 1, context);
    for (const auto context : {37u, 137u, 1053u})
      run(runtime, arena, 32, 8, 64, 1, context);
    for (const auto geometry : {std::pair{8u, 8u}, std::pair{16u, 1u}, std::pair{12u, 4u}})
      run(runtime, arena, geometry.first, geometry.second, 128, 1, 137);
    run(runtime, arena, 56, 8, 128, 1, 4096, false, 18);
    run(runtime, arena, 56, 8, 128, 1, 137, true);
    std::cout << "paged GQA frozen-input numerical and tail gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
