// SPDX-License-Identifier: Apache-2.0
// Habana Gaudi2 TPC Simulator verification for 1K FP16 + 2K FP8_143 hybrid prefill kernels:
// - gk_fp8_act_row_scale_f32_v1 & gk_fp8_weight_blk_scale_f32_v1
// - gk_fp8_quant_act_fp16_1k_v1 & gk_fp8_quant_act_fp8_2k_v1
//   (including all-zero group and P-NUM-41 tiny-positive 1e-15/1e-12 groups + 2^-9 subnormals)
// - gk_fp8_dequant_weight_fp16_1k_v1 & gk_fp8_dequant_weight_fp8_2k_v1
// - gk_fp8_hybrid_epilogue_bf16_v1 & gk_fp8_hybrid_epilogue_bias_bf16_v1
// - End-to-end 1K FP16 + 2K FP8_143 (with Gaudi2 MME FP8 subnormal FTZ) vs independent FP64
//   original OCP W8A8 reference on high-cancellation K=7168, K=384 (odd group tail + bias),
//   K=22016, and tiny-scale matrices (<= 0.0005 << 0.003 gate).
#include <VPEStats.h>
#include <tpc_kernel_lib_interface.h>
#include <tpc_test_core_api.h>

#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tpc_lib_api;

namespace {

uint16_t bf16_rne(float v) {
  if (v == 0.0f) return 0x0000U;
  const uint32_t bits = std::bit_cast<uint32_t>(v);
  if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
  return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

double direct_bf16_f64(double v) {
  if (!std::isfinite(v) || v == 0.0) return v;
  const uint64_t bits = std::bit_cast<uint64_t>(v);
  const int exp = int((bits >> 52) & 0x7ffULL) - 1023;
  if (exp < -126) {
    const double scaled = std::ldexp(std::abs(v), 133);
    const double fl = std::floor(scaled);
    const double diff = scaled - fl;
    uint64_t q = static_cast<uint64_t>(fl);
    if (diff > 0.5 || (diff == 0.5 && (q & 1ULL))) ++q;
    return std::copysign(std::ldexp(double(q), -133), v);
  }
  const uint64_t lsb = (bits >> 45) & 1ULL;
  const uint64_t guard = (bits >> 44) & 1ULL;
  const uint64_t sticky = (bits & ((1ULL << 44) - 1ULL)) != 0 ? 1ULL : 0ULL;
  const uint64_t rounded = (bits & ~((1ULL << 45) - 1ULL)) +
                           ((guard && (sticky || lsb)) ? (1ULL << 45) : 0ULL);
  return std::bit_cast<double>(rounded);
}

float widen_bf16(uint16_t b) {
  return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

float widen_fp16(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1U;
  const uint32_t exp = (h >> 10) & 0x1fU;
  const uint32_t man = h & 0x3ffU;
  if (exp == 0x1fU) {
    return std::bit_cast<float>((sign << 31) | 0x7f800000U | (man << 13));
  }
  const float val = exp ? std::ldexp(1.0f + float(man) * (1.0f / 1024.0f), int(exp) - 15)
                        : std::ldexp(float(man), -24);
  return sign ? -val : val;
}

// Gaudi2 MME syn_type_fp8_143 (expBias=7) flushes subnormals (exp == 0) to zero (P-NUM-01).
float decode_gaudi2_mme_fp8_143(uint8_t code) {
  const unsigned exp = (code >> 3) & 0xfU;
  const unsigned man = code & 7U;
  if (exp == 0U) return 0.0f;
  if (exp == 15U) return std::numeric_limits<float>::quiet_NaN();
  const float val = std::ldexp(1.0f + float(man) * 0.125f, int(exp) - 7);
  return (code & 0x80U) ? -val : val;
}

double decode_ocp_e4m3fn(uint8_t code) {
  const unsigned mag = code & 0x7fU;
  if (mag == 0x7fU) throw std::invalid_argument("OCP NaN code");
  const unsigned exp = mag >> 3;
  const unsigned man = mag & 7U;
  const double val = exp ? std::ldexp(1.0 + double(man) / 8.0, int(exp) - 7)
                         : std::ldexp(double(man), -9);
  return (code & 0x80U) ? -val : val;
}

uint8_t ocp_to_gaudi2_half_rne(uint8_t code) {
  const unsigned mag = code & 0x7fU;
  if (mag == 0x7fU) throw std::invalid_argument("OCP NaN");
  const unsigned half = mag >= 16U ? (mag - 8U)
                                   : ((mag >> 1) + ((mag & 1U) & ((mag >> 1) & 1U)));
  return static_cast<uint8_t>((code & 0x80U) | half);
}

Tensor make_tensor(TensorDataType dtype, std::initializer_list<uint64_t> sizes) {
  Tensor t{};
  t.geometry.dataType = dtype;
  t.geometry.dims = sizes.size();
  unsigned d = 0;
  for (uint64_t s : sizes) {
    t.geometry.minSizes[d] = t.geometry.maxSizes[d] = s;
    ++d;
  }
  for (unsigned i = d; i < MAX_TENSOR_DIM; ++i) {
    t.geometry.minSizes[i] = t.geometry.maxSizes[i] = 1;
  }
  return t;
}

tpc_tests::TensorDesc2 make_desc(void* data, TensorDataType dtype,
                                 std::initializer_list<uint64_t> sizes) {
  tpc_tests::TensorDesc2 desc{};
  desc.baseAddrUnion.baseAddr = reinterpret_cast<uint64_t>(data);
  desc.paddingValue = 0;
  desc.pref_stride = 0x00ba0000U;
  uint32_t elem_size_type =
      dtype == DATA_F8_143 ? 0 : ((dtype == DATA_BF16 || dtype == DATA_F16) ? 1 : 2);
  std::array<uint64_t, DEF_NUM_DIMS_IN_IRF> full_sizes{1, 1, 1, 1, 1};
  unsigned idx = 0;
  for (uint64_t s : sizes) full_sizes.at(idx++) = s;
  uint32_t last_dim = 0;
  for (unsigned d = 0; d < idx; ++d) {
    if (full_sizes[d] > 1) last_dim = d;
  }
  uint64_t stride = 1;
  for (unsigned d = 0; d < DEF_NUM_DIMS_IN_IRF; ++d) {
    desc.dimDescriptors[d].size = static_cast<uint32_t>(full_sizes[d]);
    desc.dimDescriptors[d].stride = static_cast<uint32_t>(stride);
    desc.size_stride_high[d] =
        static_cast<uint32_t>(((full_sizes[d] >> 32) & 0xfffULL) << 16) |
        static_cast<uint32_t>((stride >> 32) & 0xfffULL);
    stride *= full_sizes[d];
  }
  desc.configuration = (last_dim << 16) | 0x1f00U | elem_size_type;
  return desc;
}

struct Provider {
  void* handle = nullptr;
  pfnInstantiateTpcKernel instantiate = nullptr;
  explicit Provider(const char* path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
    instantiate = reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(handle, "InstantiateTpcKernel"));
    if (!instantiate) throw std::runtime_error("missing InstantiateTpcKernel");
  }
  ~Provider() {
    if (handle) dlclose(handle);
  }
  void run(const char* guid, std::vector<Tensor> inputs, std::vector<Tensor> outputs,
           std::vector<tpc_tests::TensorDesc2>& descs, const int32_t* tile_tag = nullptr) const {
    HabanaKernelParams params{};
    params.deviceId = DEVICE_ID_GAUDI2;
    params.apiVersion = 1;
    std::strcpy(params.guid.name, guid);
    params.inputTensorNr = inputs.size();
    params.outputTensorNr = outputs.size();
    params.inputTensors = inputs.data();
    params.outputTensors = outputs.data();
    if (tile_tag) {
      params.nodeParams.nodeParams = const_cast<int32_t*>(tile_tag);
      params.nodeParams.nodeParamsSize = sizeof(*tile_tag);
    }
    std::vector<TensorAccessPattern> in_ap(inputs.size()), out_ap(outputs.size());
    HabanaKernelInstantiation inst{};
    inst.inputTensorAccessPattern = in_ap.data();
    inst.outputTensorAccessPattern = out_ap.data();
    if (instantiate(&params, &inst) != GLUE_INSUFFICIENT_ELF_BUFFER || !inst.kernel.elfSize)
      throw std::runtime_error(std::string("ELF size query failed for ") + guid);
    std::vector<uint8_t> elf(inst.kernel.elfSize);
    inst.kernel.kernelElf = elf.data();
    if (instantiate(&params, &inst) != GLUE_SUCCESS)
      throw std::runtime_error(std::string("Instantiation failed for ") + guid);
    VPEStats stats{};
    const unsigned cycles = tpc_tests::RunSimulation(
        params, inst, descs, stats, tpc_tests::e_accessPatternDefaultMode);
    if (cycles == 0 || stats.instructionsExecuted == 0)
      throw std::runtime_error(std::string("RunSimulation failed for ") + guid);
  }
};

void run_hybrid_gemm_case(const Provider& provider, uint64_t m, uint64_t n,
                          uint64_t k, bool with_bias, bool tiny_only = false) {
  const uint64_t groups = k / 128;
  const uint64_t blocks = n / 128;
  std::mt19937 rng(static_cast<uint32_t>(k * 13 + m * 7 + n + (with_bias ? 99 : 0)));
  std::normal_distribution<float> norm(0.0f, 1.0f);

  std::vector<uint16_t> x(m * k, 0);
  for (uint64_t r = 0; r < m; ++r) {
    if (!tiny_only && r == 1) continue;  // entire all-zero row guard
    for (uint64_t g = 0; g < groups; ++g) {
      if (!tiny_only && r == 0 && g == 0) continue;  // all-zero group guard
      const float env = tiny_only ? (g & 1 ? 1.0e-15f : 1.0e-12f)
                                  : (((g / 2) % 4 == 0) ? 12.0f : 1.5f);
      for (uint64_t i = 0; i < 128; ++i) {
        x[r * k + g * 128 + i] = bf16_rne(norm(rng) * env);
      }
    }
  }

  std::vector<uint8_t> w_ocp(n * k), w_prep(groups * n * 128);
  std::vector<float> orig_scales(blocks * groups), prep_scales(blocks * groups);
  for (uint64_t nb = 0; nb < blocks; ++nb) {
    for (uint64_t g = 0; g < groups; ++g) {
      orig_scales[nb * groups + g] = (0.5f + 0.25f * float((nb * 5 + g * 3) % 11)) * 1e-3f;
      prep_scales[nb * groups + g] = orig_scales[nb * groups + g] * 2.0f;
    }
  }
  for (uint64_t col = 0; col < n; ++col) {
    for (uint64_t g = 0; g < groups; ++g) {
      for (uint64_t i = 0; i < 128; ++i) {
        uint8_t code = static_cast<uint8_t>(((col * 17 + g * 31 + i * 7) % 118) + 8);
        if ((g & 1) && (i % 5 != 0)) {
          const float prev = widen_bf16(x[(g - 1) * 128 + i]);
          const float cur = widen_bf16(x[g * 128 + i]);
          if ((prev < 0.0f) == (cur < 0.0f)) code |= 0x80U;
        } else if (((col * 3 + g * 5 + i) & 1) != 0) {
          code |= 0x80U;
        }
        w_ocp[col * k + g * 128 + i] = code;
        w_prep[(g * n + col) * 128 + i] = ocp_to_gaudi2_half_rne(code);
      }
    }
  }

  std::vector<float> bias(n, 0.0f);
  if (with_bias) {
    for (uint64_t c = 0; c < n; ++c) bias[c] = (c & 1 ? -0.125f : 0.125f) + float(c) * 1e-4f;
  }

  const int32_t tag0 = 7, tag1 = 8;
  std::vector<float> row_scale(m, -1.0f), blk_scale(blocks, -1.0f);
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(x.data(), DATA_BF16, {k, m}),
                                          make_desc(row_scale.data(), DATA_F32, {1, m})};
    provider.run("gk_fp8_act_row_scale_f32_v1",
                 {make_tensor(DATA_BF16, {k, m})}, {make_tensor(DATA_F32, {1, m})}, d, &tag0);
  }
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(prep_scales.data(), DATA_F32, {groups, blocks}),
                                          make_desc(blk_scale.data(), DATA_F32, {1, blocks})};
    provider.run("gk_fp8_weight_blk_scale_f32_v1",
                 {make_tensor(DATA_F32, {groups, blocks})}, {make_tensor(DATA_F32, {1, blocks})},
                 d, &tag0);
  }

  std::vector<uint16_t> act_1k(m * k, 0x7e00U);
  std::vector<uint8_t> act_2k(m * k * 2, 0x7fU);
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(x.data(), DATA_BF16, {k, m}),
                                          make_desc(row_scale.data(), DATA_F32, {1, m}),
                                          make_desc(act_1k.data(), DATA_F16, {k, m})};
    provider.run("gk_fp8_quant_act_fp16_1k_v1",
                 {make_tensor(DATA_BF16, {k, m}), make_tensor(DATA_F32, {1, m})},
                 {make_tensor(DATA_F16, {k, m})}, d, &tag0);
  }
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(x.data(), DATA_BF16, {k, m}),
                                          make_desc(row_scale.data(), DATA_F32, {1, m}),
                                          make_desc(act_2k.data(), DATA_F8_143, {k * 2, m})};
    provider.run("gk_fp8_quant_act_fp8_2k_v1",
                 {make_tensor(DATA_BF16, {k, m}), make_tensor(DATA_F32, {1, m})},
                 {make_tensor(DATA_F8_143, {k * 2, m})}, d, &tag1);
  }

  std::vector<uint16_t> w_1k(n * k, 0x7e00U);
  std::vector<uint8_t> w_2k(n * k * 2, 0x7fU);
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(w_prep.data(), DATA_F8_143, {128, n, groups}),
                                          make_desc(prep_scales.data(), DATA_F32, {groups, blocks}),
                                          make_desc(blk_scale.data(), DATA_F32, {1, blocks}),
                                          make_desc(w_1k.data(), DATA_F16, {k, n})};
    provider.run("gk_fp8_dequant_weight_fp16_1k_v1",
                 {make_tensor(DATA_F8_143, {128, n, groups}),
                  make_tensor(DATA_F32, {groups, blocks}),
                  make_tensor(DATA_F32, {1, blocks})},
                 {make_tensor(DATA_F16, {k, n})}, d, &tag0);
  }
  {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(w_prep.data(), DATA_F8_143, {128, n, groups}),
                                          make_desc(prep_scales.data(), DATA_F32, {groups, blocks}),
                                          make_desc(blk_scale.data(), DATA_F32, {1, blocks}),
                                          make_desc(w_2k.data(), DATA_F8_143, {k * 2, n})};
    provider.run("gk_fp8_dequant_weight_fp8_2k_v1",
                 {make_tensor(DATA_F8_143, {128, n, groups}),
                  make_tensor(DATA_F32, {groups, blocks}),
                  make_tensor(DATA_F32, {1, blocks})},
                 {make_tensor(DATA_F8_143, {k * 2, n})}, d, &tag1);
  }

  std::vector<float> c0(m * n, 0.0f), c1(m * n, 0.0f);
  for (uint64_t r = 0; r < m; ++r) {
    for (uint64_t col = 0; col < n; ++col) {
      float s0 = 0.0f, s1 = 0.0f;
      for (uint64_t inner = 0; inner < k; ++inner) {
        s0 += widen_fp16(act_1k[r * k + inner]) * widen_fp16(w_1k[col * k + inner]);
      }
      for (uint64_t inner = 0; inner < k * 2; ++inner) {
        s1 += decode_gaudi2_mme_fp8_143(act_2k[r * k * 2 + inner]) *
              decode_gaudi2_mme_fp8_143(w_2k[col * k * 2 + inner]);
      }
      c0[r * n + col] = s0;
      c1[r * n + col] = s1;
    }
  }

  std::vector<uint16_t> out_bf16(m * n, 0x7fc0U);
  if (with_bias) {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(c0.data(), DATA_F32, {n, m}),
                                          make_desc(c1.data(), DATA_F32, {n, m}),
                                          make_desc(row_scale.data(), DATA_F32, {1, m}),
                                          make_desc(blk_scale.data(), DATA_F32, {1, blocks}),
                                          make_desc(bias.data(), DATA_F32, {n}),
                                          make_desc(out_bf16.data(), DATA_BF16, {n, m})};
    provider.run("gk_fp8_hybrid_epilogue_bias_bf16_v1",
                 {make_tensor(DATA_F32, {n, m}), make_tensor(DATA_F32, {n, m}),
                  make_tensor(DATA_F32, {1, m}), make_tensor(DATA_F32, {1, blocks}),
                  make_tensor(DATA_F32, {n})},
                 {make_tensor(DATA_BF16, {n, m})}, d);
  } else {
    std::vector<tpc_tests::TensorDesc2> d{make_desc(c0.data(), DATA_F32, {n, m}),
                                          make_desc(c1.data(), DATA_F32, {n, m}),
                                          make_desc(row_scale.data(), DATA_F32, {1, m}),
                                          make_desc(blk_scale.data(), DATA_F32, {1, blocks}),
                                          make_desc(out_bf16.data(), DATA_BF16, {n, m})};
    provider.run("gk_fp8_hybrid_epilogue_bf16_v1",
                 {make_tensor(DATA_F32, {n, m}), make_tensor(DATA_F32, {n, m}),
                  make_tensor(DATA_F32, {1, m}), make_tensor(DATA_F32, {1, blocks})},
                 {make_tensor(DATA_BF16, {n, m})}, d);
  }

  std::array<double, 127> magnitudes{};
  for (unsigned c = 0; c < magnitudes.size(); ++c) magnitudes[c] = decode_ocp_e4m3fn(c);
  std::vector<double> q_x(m * k);
  for (uint64_t r = 0; r < m; ++r) {
    for (uint64_t g = 0; g < groups; ++g) {
      float max_abs = 0.0f;
      for (uint64_t i = 0; i < 128; ++i)
        max_abs = std::max(max_abs, std::abs(widen_bf16(x[r * k + g * 128 + i])));
      if (max_abs == 0.0f) max_abs = 1.0f;
      const float act_scale_f32 = max_abs / 448.0f;
      const double act_scale = double(act_scale_f32);
      for (uint64_t i = 0; i < 128; ++i) {
        const float val_f32 = widen_bf16(x[r * k + g * 128 + i]);
        const double val = double(val_f32);
        const double scaled = std::min(448.0, double(std::abs(val_f32) / act_scale_f32));
        auto up_it = std::lower_bound(magnitudes.begin(), magnitudes.end(), scaled);
        unsigned code = up_it == magnitudes.end() ? 126U : unsigned(up_it - magnitudes.begin());
        if (code > 0) {
          const double down = scaled - magnitudes[code - 1];
          const double up = magnitudes[code] - scaled;
          if (down < up || (down == up && ((code - 1) & 1U) == 0)) --code;
        }
        q_x[r * k + g * 128 + i] = std::copysign(magnitudes[code] * act_scale, val);
      }
    }
  }

  double max_rel_hybrid = 0.0, max_rel_1k = 0.0, max_rel_f32 = 0.0;
  for (uint64_t r = 0; r < m; ++r) {
    double err_hybrid = 0.0, err_1k = 0.0, err_f32 = 0.0, ref_sq = 0.0, ref_f64_sq = 0.0;
    for (uint64_t col = 0; col < n; ++col) {
      double ref = with_bias ? double(bias[col]) : 0.0;
      for (uint64_t inner = 0; inner < k; ++inner) {
        const double w = decode_ocp_e4m3fn(w_ocp[col * k + inner]) *
                         double(orig_scales[(col / 128) * groups + inner / 128]);
        ref += q_x[r * k + inner] * w;
      }
      const double ref_bf16 = direct_bf16_f64(ref);
      const double act_h = widen_bf16(out_bf16[r * n + col]);
      const float c0_scaled = (c0[r * n + col] * row_scale[r]) * blk_scale[col / 128];
      const float comb_f32 =
          ((c0[r * n + col] + 4.0f * c1[r * n + col]) * row_scale[r]) * blk_scale[col / 128] +
          (with_bias ? bias[col] : 0.0f);
      const double act_1 = widen_bf16(bf16_rne(c0_scaled + (with_bias ? bias[col] : 0.0f)));
      if (!std::isfinite(act_h)) throw std::runtime_error("nonfinite hybrid output");
      err_hybrid += (act_h - ref_bf16) * (act_h - ref_bf16);
      err_1k += (act_1 - ref_bf16) * (act_1 - ref_bf16);
      err_f32 += (double(comb_f32) - ref) * (double(comb_f32) - ref);
      ref_sq += ref_bf16 * ref_bf16;
      ref_f64_sq += ref * ref;
    }
    if (ref_sq == 0.0) {
      if (err_hybrid != 0.0) throw std::runtime_error("nonzero error on identically zero row");
      continue;
    }
    max_rel_hybrid = std::max(max_rel_hybrid, std::sqrt(err_hybrid / ref_sq));
    max_rel_1k = std::max(max_rel_1k, std::sqrt(err_1k / ref_sq));
    max_rel_f32 = std::max(max_rel_f32, std::sqrt(err_f32 / ref_f64_sq));
  }
  std::cout << "[PASS] 1K FP16 + 2K FP8_143 tpcsim (M=" << m << ", N=" << n << ", K=" << k
            << ", bias=" << with_bias << ", tiny=" << tiny_only
            << "): pre_bf16_f32_rel_l2=" << max_rel_f32
            << ", 1K_FP16_bf16_rel_l2=" << max_rel_1k
            << " -> hybrid_bf16_rel_l2=" << max_rel_hybrid << '\n';
  if (!(max_rel_f32 <= 1.0e-4 && max_rel_hybrid <= 0.001))
    throw std::runtime_error("1K FP16 + 2K FP8_143 exceeded simulator accuracy gates");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: fp8_hybrid_tpcsim_test <libgaudi_kernels_bundle.so>");
    Provider provider(argv[1]);
    run_hybrid_gemm_case(provider, 4, 256, 7168, false, false);
    run_hybrid_gemm_case(provider, 37, 256, 384, true, false);
    run_hybrid_gemm_case(provider, 2, 128, 22016, false, false);
    run_hybrid_gemm_case(provider, 4, 128, 256, false, true);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
