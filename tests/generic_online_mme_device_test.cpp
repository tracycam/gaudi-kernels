// SPDX-License-Identifier: Apache-2.0
// Two fixed MME tiles, device online reduction, and independent CPU references.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace gaudirt::runtime;
namespace {
void check(synStatus s, const char* stage) {
  if (s != synSuccess) throw std::runtime_error(std::string(stage) + " status " + std::to_string(int(s)));
}
uint16_t to_bf16(float f) {
  auto bits = std::bit_cast<uint32_t>(f);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float widen(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
uint16_t direct_bf16(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
  auto bits = std::bit_cast<uint64_t>(value);
  uint16_t sign = (bits >> 48) & 0x8000;
  uint64_t exponent = (bits >> 52) & 0x7ff, fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent < 897) return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(std::abs(value), 133)));
  if (exponent > 1150) return sign | 0x7f80;
  uint64_t payload = ((exponent - 896) << 7) | (fraction >> 45);
  uint64_t tail = fraction & ((uint64_t{1} << 45) - 1), half = uint64_t{1} << 44;
  payload += tail > half || (tail == half && (payload & 1));
  return sign | static_cast<uint16_t>(payload);
}
struct Builder {
  synGraphHandle graph{};
  std::vector<synTensor> tensors;
  std::vector<synSectionHandle> sections;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> launch;
  std::string prefix;
  explicit Builder(std::string name) : prefix(std::move(name)) {
    tensors.reserve(64);
    sections.reserve(64);
    names.reserve(64);
    launch.reserve(32);
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph");
  }
  size_t add(std::vector<uint64_t> dims, synDataType dtype, uint64_t addr = 0) {
    const size_t id = tensors.size();
    names.push_back(prefix + "_t" + std::to_string(id));
    sections.push_back(nullptr);
    tensors.push_back(nullptr);
    synTensorDescriptor desc{};
    desc.m_name = names.back().c_str();
    desc.m_dataType = dtype;
    desc.m_tensorType = DATA_TENSOR;
    desc.m_dims = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) desc.m_sizes[i] = desc.m_minSizes[i] = dims[i];
    if (addr) {
      check(synSectionCreate(&sections.back(), 0, graph), "section");
      check(synSectionSetPersistent(sections.back(), true), "persistent");
    }
    check(synTensorCreate(&tensors.back(), &desc, sections.back(), 0), "tensor");
    if (addr) {
      synLaunchTensorInfo x{};
      x.tensorName = names.back().c_str();
      x.tensorType = DATA_TENSOR;
      x.pTensorAddress = addr;
      for (size_t i = 0; i < dims.size(); ++i) x.tensorSize[i] = dims[i];
      launch.push_back(x);
    }
    return id;
  }
  void node(const char* op, std::initializer_list<size_t> ins, std::initializer_list<size_t> outs,
            const void* params = nullptr, size_t param_bytes = 0) {
    std::vector<synTensor> a, b;
    for (auto i : ins) a.push_back(tensors.at(i));
    for (auto i : outs) b.push_back(tensors.at(i));
    const auto name = prefix + "_n" + std::to_string(node_count++);
    check(synNodeCreate(graph, a.data(), b.data(), a.size(), b.size(), const_cast<void*>(params), param_bytes,
                        op, name.c_str(), nullptr, nullptr),
          "node");
  }
  void destroy() {
    check(synGraphDestroy(graph), "graph destroy");
    for (auto t : tensors) check(synTensorDestroy(t), "tensor destroy");
    for (auto s : sections)
      if (s) check(synSectionDestroy(s), "section destroy");
  }
  size_t node_count = 0;
};
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 10)
      throw std::invalid_argument(
          "expected cache mode output d q_bucket valid_rows kv_tile q_heads kv_heads");
    const std::string mode = argv[2];
    const uint32_t d = std::stoul(argv[4]), rows = std::stoul(argv[5]), valid = std::stoul(argv[6]);
    const uint32_t kv = std::stoul(argv[7]), heads = std::stoul(argv[8]), kvh = std::stoul(argv[9]);
    if ((mode != "cold" && mode != "warm") || (d != 64 && d != 128) || !rows || rows > 256 || !valid ||
        valid > rows || !kv || kv > 2048 || kv % 128 || !kvh || heads % kvh || heads > 256 ||
        static_cast<uint64_t>(kv) * rows * heads > 4ULL * 1024 * 1024)
      throw std::invalid_argument("unsupported geometry");
    const uint32_t group = heads / kvh;
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
    std::ofstream bytes(argv[3], std::ios::binary);
    if (!bytes) throw std::runtime_error("cannot open output");
    std::mt19937 generator(20261106);
    std::normal_distribution<float> normal(0.f, 0.6f);
    std::vector<uint16_t> q(d * rows * heads);
    std::array<std::vector<uint16_t>, 2> k{std::vector<uint16_t>(d * kv * kvh),
                                           std::vector<uint16_t>(d * kv * kvh)};
    std::array<std::vector<uint16_t>, 2> v{k[0], k[0]};
    for (auto& x : q) x = to_bf16(normal(generator));
    for (auto& tile : k)
      for (auto& x : tile) x = to_bf16(normal(generator));
    for (auto& tile : v)
      for (auto& x : tile) x = to_bf16(normal(generator));
    std::vector<int32_t> positions(rows);
    for (uint32_t row = 0; row < rows; ++row)
      positions[row] = static_cast<int32_t>(rows == 1 ? 2 * kv - 8 : kv + row);
    std::array<std::array<int32_t, 3>, 2> header{
        {{static_cast<int32_t>(valid), static_cast<int32_t>(kv), 0},
         {static_cast<int32_t>(valid), static_cast<int32_t>(kv - 7), static_cast<int32_t>(kv)}}};
    std::vector<uint16_t> output(d * rows * heads);
    std::vector<float> maxima(rows * heads), sums(rows * heads);
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 256ULL * 1024 * 1024);
    RecipeCache cache(argv[1]);
    const auto q_addr = arena.allocate(q.size() * 2);
    std::array<uint64_t, 2> k_addr{}, v_addr{}, header_addr{};
    for (unsigned i = 0; i < 2; ++i) {
      k_addr[i] = arena.allocate(k[i].size() * 2);
      v_addr[i] = arena.allocate(v[i].size() * 2);
      header_addr[i] = arena.allocate(sizeof(header[i]));
    }
    const auto positions_addr = arena.allocate(positions.size() * 4);
    const auto output_addr = arena.allocate(output.size() * 2), max_addr = arena.allocate(maxima.size() * 4),
               sum_addr = arena.allocate(sums.size() * 4);
    arena.copy_h2d_sync(runtime.h2d_stream(), q.data(), q_addr, q.size() * 2);
    for (unsigned i = 0; i < 2; ++i) {
      arena.copy_h2d_sync(runtime.h2d_stream(), k[i].data(), k_addr[i], k[i].size() * 2);
      arena.copy_h2d_sync(runtime.h2d_stream(), v[i].data(), v_addr[i], v[i].size() * 2);
    }
    arena.copy_h2d_sync(runtime.h2d_stream(), positions.data(), positions_addr, positions.size() * 4);
    const std::string prefix = "mme_online_d" + std::to_string(d) + "_q" + std::to_string(rows) + "_k" +
                               std::to_string(kv) + "_h" + std::to_string(heads) + "_kh" +
                               std::to_string(kvh);
    Builder b(prefix);
    const auto tq = b.add({d, rows, group, kvh}, syn_type_bf16, q_addr);
    const auto tp = b.add({rows}, syn_type_int32, positions_addr);
    const auto tout = b.add({d, rows, heads}, syn_type_bf16, output_addr);
    const auto tm = b.add({rows, heads}, syn_type_float, max_addr),
               tl = b.add({rows, heads}, syn_type_float, sum_addr);
    const synGEMMParams qk{false, true}, pv{false, false};
    std::array<size_t, 2> m{}, l{}, o{};
    for (unsigned i = 0; i < 2; ++i) {
      auto tk = b.add({d, kv, 1, kvh}, syn_type_bf16, k_addr[i]),
           tv = b.add({d, kv, 1, kvh}, syn_type_bf16, v_addr[i]);
      auto th = b.add({3}, syn_type_int32, header_addr[i]);
      auto score4 = b.add({kv, rows, group, kvh}, syn_type_float),
           score3 = b.add({kv, rows, heads}, syn_type_float);
      auto p3 = b.add({kv, rows, heads}, syn_type_bf16), p4 = b.add({kv, rows, group, kvh}, syn_type_bf16);
      m[i] = b.add({rows, heads}, syn_type_float);
      l[i] = b.add({rows, heads}, syn_type_float);
      auto o4 = b.add({d, rows, group, kvh}, syn_type_float);
      o[i] = b.add({d, rows, heads}, syn_type_float);
      b.node("batch_gemm", {tq, tk}, {score4}, &qk, sizeof(qk));
      b.node("reshape", {score4}, {score3});
      b.node("gk_attn_tile_softmax_bf16_v1", {score3, tp, th}, {p3, m[i], l[i]}, &scale, sizeof(scale));
      b.node("reshape", {p3}, {p4});
      b.node("batch_gemm", {p4, tv}, {o4}, &pv, sizeof(pv));
      b.node("reshape", {o4}, {o[i]});
    }
    auto merged = b.add({d, rows, heads}, syn_type_float);
    b.node("gk_attn_online_merge_f32_v1", {m[0], l[0], o[0], m[1], l[1], o[1]}, {tm, tl, merged});
    b.node("gk_attn_online_finalize_bf16_v1", {tl, merged}, {tout});
    auto recipe = compile_cached(b.graph, prefix, prefix, &cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 1}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{1, 0}))
      throw std::runtime_error("cache counts");
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe), "workspace");
    initialize_launch_tensor_ids(recipe, b.launch.data(), b.launch.size());
    for (int pass = 0; pass < 4; ++pass) {
      header[0][2] = (pass == 1 || pass == 2) ? static_cast<int32_t>(4 * kv) : 0;
      header[1][2] = (pass == 1 || pass == 3) ? static_cast<int32_t>(4 * kv)
                                              : (pass == 2 ? 0 : static_cast<int32_t>(kv));
      for (unsigned i = 0; i < 2; ++i)
        arena.copy_h2d_sync(runtime.h2d_stream(), header[i].data(), header_addr[i], sizeof(header[i]));
      check(synLaunch(runtime.compute_stream(), b.launch.data(), b.launch.size(),
                      arena.ensure_workspace(workspace), recipe, 0),
            "launch");
      runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(), output_addr, output.data(), output.size() * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), max_addr, maxima.data(), maxima.size() * 4);
      arena.copy_d2h_sync(runtime.d2h_stream(), sum_addr, sums.data(), sums.size() * 4);
      bytes.write(reinterpret_cast<const char*>(output.data()), output.size() * 2);
      bytes.write(reinterpret_cast<const char*>(maxima.data()), maxima.size() * 4);
      bytes.write(reinterpret_cast<const char*>(sums.data()), sums.size() * 4);
      if (!bytes) throw std::runtime_error("diagnostic output write");
      double max64_error = 0, max_tile_error = 0, max_sum_error = 0;
      for (uint32_t row = 0; row < rows; ++row) {
        double e64 = 0, etile = 0, n64 = 0, ntile = 0;
        for (uint32_t head = 0; head < heads; ++head) {
          const size_t stat = static_cast<size_t>(head) * rows + row;
          std::array<int, 2> count{};
          for (unsigned tile = 0; tile < 2; ++tile)
            count[tile] = row < valid
                              ? std::max(0, std::min(header[tile][1], positions[row] - header[tile][2] + 1))
                              : 0;
          if (!count[0] && !count[1]) {
            if (!std::isinf(maxima[stat]) || maxima[stat] >= 0 || sums[stat] != 0)
              throw std::runtime_error("empty tile stats");
            for (uint32_t col = 0; col < d; ++col)
              if (output[stat * d + col] != 0) throw std::runtime_error("empty tile output");
            continue;
          }
          std::array<std::vector<double>, 2> scores64, prob64;
          std::array<std::vector<float>, 2> scores32, prob32;
          double global64 = -std::numeric_limits<double>::infinity();
          std::array<float, 2> m32{-std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()},
              sum32{};
          for (unsigned tile = 0; tile < 2; ++tile) {
            scores64[tile].resize(count[tile]);
            scores32[tile].resize(count[tile]);
            prob64[tile].resize(count[tile]);
            prob32[tile].resize(count[tile]);
            for (int t = 0; t < count[tile]; ++t) {
              double a = 0;
              float c = 0;
              for (uint32_t col = 0; col < d; ++col) {
                auto x = widen(q[stat * d + col]);
                auto y = widen(k[tile][(static_cast<size_t>(head / group) * kv + t) * d + col]);
                a += static_cast<double>(x) * y;
                c += x * y;
              }
              scores64[tile][t] = a * scale;
              scores32[tile][t] = c * scale;
              global64 = std::max(global64, scores64[tile][t]);
              m32[tile] = std::max(m32[tile], scores32[tile][t]);
            }
          }
          double den64 = 0;
          for (unsigned tile = 0; tile < 2; ++tile)
            for (int t = 0; t < count[tile]; ++t) {
              prob64[tile][t] = std::exp(scores64[tile][t] - global64);
              den64 += prob64[tile][t];
              prob32[tile][t] = std::exp(scores32[tile][t] - m32[tile]);
              sum32[tile] += prob32[tile][t];
            }
          const float merged_max = std::max(m32[0], m32[1]);
          std::array<float, 2> factor{};
          for (unsigned tile = 0; tile < 2; ++tile)
            if (count[tile]) factor[tile] = std::exp(m32[tile] - merged_max);
          const float den32 = sum32[0] * factor[0] + sum32[1] * factor[1];
          if (!std::isfinite(sums[stat]) || sums[stat] <= 0 || !std::isfinite(maxima[stat]))
            throw std::runtime_error("nonfinite stats");
          max_sum_error = std::max(max_sum_error, std::abs(static_cast<double>(sums[stat] - den32) / den32));
          for (uint32_t col = 0; col < d; ++col) {
            double z64 = 0;
            std::array<float, 2> z32{};
            for (unsigned tile = 0; tile < 2; ++tile)
              for (int t = 0; t < count[tile]; ++t) {
                auto value = widen(v[tile][(static_cast<size_t>(head / group) * kv + t) * d + col]);
                z64 += prob64[tile][t] * value;
                z32[tile] += widen(to_bf16(prob32[tile][t])) * value;
              }
            const double actual = widen(output[stat * d + col]);
            const double reference = widen(direct_bf16(z64 / den64));
            const double tiled = widen(to_bf16((z32[0] * factor[0] + z32[1] * factor[1]) / den32));
            if (!std::isfinite(actual)) throw std::runtime_error("nonfinite output");
            e64 += (actual - reference) * (actual - reference);
            n64 += reference * reference;
            etile += (actual - tiled) * (actual - tiled);
            ntile += tiled * tiled;
          }
        }
        if (n64) max64_error = std::max(max64_error, std::sqrt(e64 / n64));
        if (ntile) max_tile_error = std::max(max_tile_error, std::sqrt(etile / ntile));
      }
      std::cout << "pass " << pass << " fp64 " << max64_error << " tiled " << max_tile_error << " sum_rel "
                << max_sum_error << '\n';
      if (max64_error > 0.01 || max_tile_error > 0.001 || max_sum_error > 1e-4)
        throw std::runtime_error("numeric gate");
    }
    std::cout << "cold/warm counts " << counts.first << '/' << counts.second << " workspace " << workspace
              << " bytes\n";
    check(synRecipeDestroy(recipe), "recipe destroy");
    b.destroy();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
