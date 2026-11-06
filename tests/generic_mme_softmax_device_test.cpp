// FP64 and valid-row audit of the single-output attention softmax.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

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

uint16_t direct_bf16(double value) {
  if (!std::isfinite(value) || value < 0) throw std::runtime_error("invalid reference value");
  if (value == 0) return 0;
  const uint64_t bits = std::bit_cast<uint64_t>(value);
  const uint64_t exponent = (bits >> 52) & 0x7ff;
  const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent < 897) return static_cast<uint16_t>(std::nearbyint(std::ldexp(value, 133)));
  uint64_t payload = ((exponent - 896) << 7) | (fraction >> 45);
  const auto tail = fraction & ((uint64_t{1} << 45) - 1);
  const auto half = uint64_t{1} << 44;
  payload += tail > half || (tail == half && (payload & 1));
  return static_cast<uint16_t>(payload);
}

uint16_t round_fp32(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

float widen(uint16_t value) { return std::bit_cast<float>(uint32_t{value} << 16); }
}

int main() {
  try {
    constexpr uint32_t context = 137, ratio = 7, kv_heads = 8, bucket = 2;
    constexpr uint32_t groups = kv_heads * bucket, valid_length = 92;
    constexpr uint16_t sentinel = 0x3e80;
    const size_t count = size_t{context} * ratio * groups;
    std::vector<float> scores(count);
    std::vector<uint16_t> output(count, sentinel);
    for (size_t i = 0; i < count; ++i)
      scores[i] = static_cast<float>(static_cast<int>((i * 197 + 31) % 513) - 256) / 64.f;
    const std::array<int32_t, bucket> positions{valid_length - 1, 0};
    const int32_t valid_rows = 1;
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 8 * 1024 * 1024);
    const std::array<uint64_t, 4> addresses{arena.allocate(count * sizeof(float)),
        arena.allocate(sizeof(positions)), arena.allocate(sizeof(valid_rows)), arena.allocate(count * sizeof(uint16_t))};
    arena.copy_h2d_sync(runtime.h2d_stream(), scores.data(), addresses[0], count * sizeof(float));
    arena.copy_h2d_sync(runtime.h2d_stream(), positions.data(), addresses[1], sizeof(positions));
    arena.copy_h2d_sync(runtime.h2d_stream(), &valid_rows, addresses[2], sizeof(valid_rows));
    arena.copy_h2d_sync(runtime.h2d_stream(), output.data(), addresses[3], count * sizeof(uint16_t));
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::array<synTensor, 4> tensors{};
    std::array<synSectionHandle, 4> sections{};
    std::array<synLaunchTensorInfo, 4> bindings{};
    const std::array<std::string, 4> names{"softmax_t0", "softmax_t1", "softmax_t2", "softmax_t3"};
    check(synGraphCreate(&graph, synDeviceGaudi2));
    for (size_t i = 0; i < tensors.size(); ++i) {
      check(synSectionCreate(&sections[i], 0, graph));
      check(synSectionSetPersistent(sections[i], true));
      synTensorDescriptor desc{};
      desc.m_name = names[i].c_str();
      desc.m_dataType = i == 0 ? syn_type_float : i == 3 ? syn_type_bf16 : syn_type_int32;
      desc.m_dims = i == 0 || i == 3 ? 3 : 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 || i == 3 ? context : i == 1 ? bucket : 1;
      if (desc.m_dims == 3) {
        desc.m_sizes[1] = desc.m_minSizes[1] = ratio;
        desc.m_sizes[2] = desc.m_minSizes[2] = groups;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
      bindings[i].tensorName = names[i].c_str();
      bindings[i].tensorType = DATA_TENSOR;
      bindings[i].pTensorAddress = addresses[i];
      for (unsigned dim = 0; dim < desc.m_dims; ++dim) bindings[i].tensorSize[dim] = desc.m_sizes[dim];
    }
    struct Params { int32_t heads; float scale; } params{kv_heads, 1.f};
    check(synNodeCreate(graph, tensors.data(), &tensors[3], 3, 1, &params, sizeof(params),
                        "gk_paged_gqa_mme_softmax_bf16_v1", "softmax_n0", nullptr, nullptr));
    check(synGraphCompile(&recipe, graph, "softmax_single_output_test", nullptr));
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe));
    initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
    check(synLaunch(runtime.compute_stream(), bindings.data(), bindings.size(),
                    arena.ensure_workspace(workspace), recipe, 0));
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[3], output.data(), count * sizeof(uint16_t));
    double worst = 0, worst_floor = 0;
    for (uint32_t group = 0; group < groups; ++group) {
      for (uint32_t head = 0; head < ratio; ++head) {
        const auto base = size_t{context} * (head + ratio * group);
        if (group >= kv_heads) {
          for (uint32_t token = 0; token < context; ++token)
            if (output[base + token] != sentinel) throw std::runtime_error("invalid row was written");
          continue;
        }
        const auto max = *std::max_element(scores.begin() + base, scores.begin() + base + valid_length);
        double sum64 = 0;
        float sum32 = 0;
        for (uint32_t token = 0; token < valid_length; ++token) {
          sum64 += std::exp(double{scores[base + token]} - max);
          sum32 += std::exp(scores[base + token] - max);
        }
        double error = 0, floor = 0, norm = 0;
        for (uint32_t token = 0; token < context; ++token) {
          const auto truth = token < valid_length ?
              widen(direct_bf16(std::exp(double{scores[base + token]} - max) / sum64)) : 0.f;
          const auto baseline = token < valid_length ?
              widen(round_fp32(std::exp(scores[base + token] - max) / sum32)) : 0.f;
          const auto actual = widen(output[base + token]);
          if (!std::isfinite(actual) || !std::isfinite(truth) || !std::isfinite(baseline))
            throw std::runtime_error("nonfinite softmax result");
          norm += double{truth} * truth;
          error += double{actual - truth} * (actual - truth);
          floor += double{baseline - truth} * (baseline - truth);
        }
        const auto denominator = std::max(std::sqrt(norm), 1e-30);
        error = std::sqrt(error) / denominator;
        floor = std::sqrt(floor) / denominator;
        worst = std::max(worst, error);
        worst_floor = std::max(worst_floor, floor);
        if (error > std::max(3e-3, 2 * floor) || error > 1e-2)
          throw std::runtime_error("softmax exceeds FP64 error gate");
      }
    }
    std::cout << "single BF16 probability output max_row_relative_l2 " << worst
              << " fp32_floor " << worst_floor << " invalid_rows_written 0\n";
    check(synRecipeDestroy(recipe));
    check(synGraphDestroy(graph));
    for (auto tensor : tensors) check(synTensorDestroy(tensor));
    for (auto section : sections) check(synSectionDestroy(section));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
