// SPDX-License-Identifier: Apache-2.0
// Standalone Synapse device verification suite for Track A TPC kernels & Glue.
// Strictly uses libSynapse.so + libgaudi_kernels_bundle.so (zero legacy bridge)
// and pins execution to HABANA_VISIBLE_MODULES=0 (module_id 0).

#include <synapse_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define SYN_CHECK(expr)                                                     \
  do {                                                                      \
    synStatus _st = (expr);                                                 \
    if (_st != synSuccess) {                                                \
      std::ostringstream _oss;                                              \
      _oss << #expr << " failed with synStatus=" << static_cast<int>(_st); \
      throw std::runtime_error(_oss.str());                                 \
    }                                                                       \
  } while (0)

struct BF16 {
  uint16_t bits;
};

uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);
  uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

uint32_t f32_bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

float bits_f32(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

uint32_t f32_ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isnan(a) || std::isnan(b)) return 0xffffffffu;
  if (a == b) return 0;
  if (std::isinf(a) || std::isinf(b)) return 0xffffffffu;
  auto to_ord = [](float v) -> int64_t {
    uint32_t u = f32_bits(v);
    if (u & 0x80000000u) return -static_cast<int64_t>(u & 0x7fffffffu);
    return static_cast<int64_t>(u);
  };
  int64_t d = to_ord(a) - to_ord(b);
  return static_cast<uint32_t>(d < 0 ? -d : d);
}

uint32_t bf16_ulp_diff(uint16_t a, uint16_t b) {
  float fa = bf16_to_f32(a), fb = bf16_to_f32(b);
  if (std::isnan(fa) && std::isnan(fb)) return 0;
  if (std::isnan(fa) || std::isnan(fb)) return 0xffffu;
  if (fa == fb) return 0;
  auto to_ord = [](uint16_t x) -> int32_t {
    if (x & 0x8000u) return -static_cast<int32_t>(x & 0x7fffu);
    return static_cast<int32_t>(x);
  };
  int32_t d = to_ord(a) - to_ord(b);
  return static_cast<uint32_t>(d < 0 ? -d : d);
}

size_t dtype_bytes(synDataType dt) {
  switch (dt) {
    case syn_type_bf16: return 2;
    case syn_type_single:
    case syn_type_int32: return 4;
    default: throw std::invalid_argument("unsupported synDataType");
  }
}

struct TensorSpec {
  std::string name;
  synDataType dtype;
  std::vector<uint64_t> dims;  // FCD-first: dims[0] is contiguous
  const void* init_data = nullptr;
  size_t byte_size = 0;        // optional larger alloc (e.g. sentinel tail check)
  std::string alias_of;        // share section + device buffer with earlier tensor
};

struct SynapseCtx {
  synDeviceId dev = 0;
  synStreamHandle h2d = nullptr;
  synStreamHandle compute = nullptr;
  synStreamHandle d2h = nullptr;

  SynapseCtx() {
    const char* vis = std::getenv("HABANA_VISIBLE_MODULES");
    if (!vis || std::strcmp(vis, "0") != 0) {
      throw std::runtime_error("HABANA_VISIBLE_MODULES must be strictly set to '0'");
    }
    SYN_CHECK(synInitialize());
    SYN_CHECK(synDeviceAcquireByModuleId(&dev, 0));
    SYN_CHECK(synStreamCreateGeneric(&h2d, dev, 0));
    SYN_CHECK(synStreamCreateGeneric(&compute, dev, 0));
    SYN_CHECK(synStreamCreateGeneric(&d2h, dev, 0));
  }

  ~SynapseCtx() {
    if (d2h) synStreamDestroy(d2h);
    if (compute) synStreamDestroy(compute);
    if (h2d) synStreamDestroy(h2d);
    if (dev) synDeviceRelease(dev);
    synDestroy();
  }
};

struct DevBuf {
  uint64_t dev_addr = 0;
  void* host_ptr = nullptr;
  size_t bytes = 0;
  synDeviceId dev = 0;

  DevBuf(synDeviceId d, size_t n) : bytes(std::max<size_t>(n, 256)), dev(d) {
    // Align allocations to 256B for clean HBM transfers
    bytes = (bytes + 255) & ~size_t(255);
    SYN_CHECK(synDeviceMalloc(dev, bytes, 0, 0, &dev_addr));
    SYN_CHECK(synHostMalloc(dev, bytes, 0, &host_ptr));
    std::memset(host_ptr, 0, bytes);
  }
  ~DevBuf() {
    if (host_ptr) synHostFree(dev, host_ptr, 0);
    if (dev_addr) synDeviceFree(dev, dev_addr, 0);
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
};

std::map<std::string, std::vector<uint8_t>> run_single_node(
    SynapseCtx& ctx, const std::string& guid, const std::vector<TensorSpec>& inputs,
    const std::vector<TensorSpec>& outputs, const void* params, unsigned params_size,
    uint8_t fill_sentinel = 0x5A) {
  synGraphHandle graph = nullptr;
  SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));

  std::vector<synSectionHandle> sections;
  std::vector<synTensor> syn_in, syn_out, all_tensors;
  std::map<std::string, synSectionHandle> name_to_sec;
  std::map<std::string, std::unique_ptr<DevBuf>> name_to_buf;
  std::vector<synLaunchTensorInfo> launch_info;

  auto make_tensor = [&](const TensorSpec& spec, bool is_out) -> synTensor {
    size_t elem_count = 1;
    for (auto d : spec.dims) elem_count *= d;
    size_t logical_bytes = elem_count * dtype_bytes(spec.dtype);
    size_t alloc_bytes = std::max(logical_bytes, spec.byte_size);

    synSectionHandle sec = nullptr;
    if (!spec.alias_of.empty()) {
      sec = name_to_sec.at(spec.alias_of);
    } else {
      SYN_CHECK(synSectionCreate(&sec, 0, graph));
      sections.push_back(sec);
      name_to_sec[spec.name] = sec;
      auto buf = std::make_unique<DevBuf>(ctx.dev, alloc_bytes);
      if (spec.init_data) {
        std::memcpy(buf->host_ptr, spec.init_data, logical_bytes);
        if (alloc_bytes > logical_bytes) {
          std::memset(static_cast<uint8_t*>(buf->host_ptr) + logical_bytes, fill_sentinel,
                      alloc_bytes - logical_bytes);
        }
      } else if (is_out) {
        std::memset(buf->host_ptr, fill_sentinel, alloc_bytes);
      }
      SYN_CHECK(synMemCopyAsync(ctx.h2d, reinterpret_cast<uint64_t>(buf->host_ptr), buf->bytes,
                                buf->dev_addr, HOST_TO_DRAM));
      name_to_buf.emplace(spec.name, std::move(buf));
    }

    synTensor t = nullptr;
    SYN_CHECK(synTensorHandleCreate(&t, graph, DATA_TENSOR, spec.name.c_str()));
    all_tensors.push_back(t);
    SYN_CHECK(synTensorAssignToSection(t, sec, 0));

    synTensorGeometry geom{};
    geom.dims = static_cast<uint32_t>(spec.dims.size());
    for (size_t i = 0; i < spec.dims.size(); ++i) geom.sizes[i] = spec.dims[i];
    SYN_CHECK(synTensorSetGeometry(t, &geom, synGeometryMaxSizes));

    synTensorDeviceFullLayout layout{};
    layout.deviceDataType = spec.dtype;
    uint32_t stride = static_cast<uint32_t>(dtype_bytes(spec.dtype));
    for (size_t i = 0; i < spec.dims.size(); ++i) {
      layout.strides[i] = stride;
      stride *= static_cast<uint32_t>(spec.dims[i]);
    }
    SYN_CHECK(synTensorSetDeviceFullLayout(t, &layout));

    const std::string& backing = spec.alias_of.empty() ? spec.name : spec.alias_of;
    synLaunchTensorInfo linfo{};
    linfo.tensorName = spec.name.c_str();
    linfo.pTensorAddress = name_to_buf.at(backing)->dev_addr;
    linfo.tensorType = DATA_TENSOR;
    for (size_t i = 0; i < spec.dims.size(); ++i) linfo.tensorSize[i] = spec.dims[i];
    launch_info.push_back(linfo);
    return t;
  };

  for (const auto& s : inputs) syn_in.push_back(make_tensor(s, false));
  for (const auto& s : outputs) syn_out.push_back(make_tensor(s, true));
  SYN_CHECK(synStreamSynchronize(ctx.h2d));

  SYN_CHECK(synNodeCreate(graph, syn_in.data(), syn_out.data(),
                          static_cast<uint32_t>(syn_in.size()),
                          static_cast<uint32_t>(syn_out.size()),
                          params, params_size, guid.c_str(), guid.c_str(), nullptr, nullptr));

  synRecipeHandle recipe = nullptr;
  SYN_CHECK(synGraphCompile(&recipe, graph, guid.c_str(), nullptr));
  uint64_t ws_size = 0;
  SYN_CHECK(synWorkspaceGetSize(&ws_size, recipe));
  uint64_t ws_dev = 0;
  if (ws_size > 0) SYN_CHECK(synDeviceMalloc(ctx.dev, ws_size, 0, 0, &ws_dev));

  SYN_CHECK(synLaunchWithExternalEvents(ctx.compute, launch_info.data(),
                                        static_cast<uint32_t>(launch_info.size()), ws_dev,
                                        recipe, nullptr, 0, SYN_FLAGS_TENSOR_NAME));
  SYN_CHECK(synStreamSynchronize(ctx.compute));

  std::map<std::string, std::vector<uint8_t>> results;
  for (const auto& s : outputs) {
    const std::string& backing = s.alias_of.empty() ? s.name : s.alias_of;
    auto& buf = name_to_buf.at(backing);
    SYN_CHECK(synMemCopyAsync(ctx.d2h, buf->dev_addr, buf->bytes,
                              reinterpret_cast<uint64_t>(buf->host_ptr), DRAM_TO_HOST));
    SYN_CHECK(synStreamSynchronize(ctx.d2h));
    size_t elem_count = 1;
    for (auto d : s.dims) elem_count *= d;
    size_t nbytes = std::max(elem_count * dtype_bytes(s.dtype), s.byte_size);
    std::vector<uint8_t> out(nbytes);
    std::memcpy(out.data(), buf->host_ptr, nbytes);
    results.emplace(s.name, std::move(out));
  }

  if (ws_dev) synDeviceFree(ctx.dev, ws_dev, 0);
  synRecipeDestroy(recipe);
  for (auto t : all_tensors) synTensorDestroy(t);
  for (auto sec : sections) synSectionDestroy(sec);
  synGraphDestroy(graph);
  return results;
}

double gelu_exact_ref(double x) {
  if (std::isinf(x)) return x > 0.0 ? x : 0.0;
  return 0.5 * x * std::erfc(-x * 0.70710678118654752440);
}

double gelu_tanh_ref(double x) {
  if (std::isinf(x)) return x > 0.0 ? x : 0.0;
  const double k2 = 1.59576912160573071176;
  const double u2 = k2 * (x + 0.044715 * x * x * x);
  if (u2 >= 0.0) return x / (1.0 + std::exp(-u2));
  const double e = std::exp(u2);
  return x * (e / (1.0 + e));
}

void test_ewise_and_math(SynapseCtx& ctx, std::vector<std::string>& summary) {
  const uint64_t W = 577, R = 5;
  std::vector<float> special_vals = {
      0.0f, -0.0f, 1e-6f, -1e-6f, 0.6249f, -0.6249f, 0.625f, -0.625f, 0.6251f, -0.6251f,
      1.249f, -1.249f, 1.25f, -1.25f, 1.251f, -1.251f, 3.0f, -3.0f, 10.0f, -10.0f,
      20.0f, -20.0f, 88.0f, -88.0f, 100.0f, -100.0f,
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};

  std::vector<float> in_f32(W * R);
  for (size_t i = 0; i < in_f32.size(); ++i) {
    if (i < special_vals.size()) {
      in_f32[i] = special_vals[i];
    } else {
      float t = static_cast<float>(static_cast<int>(i % 241) - 120) * 0.1f;
      in_f32[i] = t;
    }
  }
  std::vector<uint16_t> in_bf16(W * R);
  for (size_t i = 0; i < in_f32.size(); ++i) in_bf16[i] = f32_to_bf16_rne(in_f32[i]);

  struct EwiseParams { int32_t op; float alpha; float beta; };
  uint32_t max_tanh_ulp = 0, max_sig_ulp = 0, max_gelu_erf_ulp = 0, max_gelu_tanh_ulp = 0;

  for (int32_t op = 0; op <= 11; ++op) {
    EwiseParams p{op, (op == 6 ? 50.0f : -2.5f), 3.5f};
    std::vector<float> f32_input = in_f32;
    if (op == 8 || op == 9) {
      for (float& v : f32_input) v = std::abs(v) + 0.01f;
    }
    auto res = run_single_node(
        ctx, "gk_ewise_unary_f32_v1",
        {{"in", syn_type_single, {W, R}, f32_input.data()}},
        {{"out", syn_type_single, {W, R}, nullptr, (W * R + 64) * 4}},
        &p, sizeof(p), 0x5A);
    const auto& bytes = res.at("out");
    for (size_t b = W * R * 4; b < bytes.size(); ++b) {
      if (bytes[b] != 0x5A) throw std::runtime_error("ewise_unary_f32 OOB tail write");
    }
    const float* got = reinterpret_cast<const float*>(bytes.data());
    uint32_t max_ulp = 0;
    for (size_t i = 0; i < W * R; ++i) {
      double x = static_cast<double>(f32_input[i]);
      double ref = 0.0;
      switch (op) {
        case 0: ref = std::exp(x); break;
        case 1: ref = std::tanh(x); break;
        case 2: ref = 1.0 / (1.0 + std::exp(-x)); break;
        case 3: ref = (x <= -88.0) ? 0.0 : x / (1.0 + std::exp(-x)); break;
        case 4: ref = gelu_exact_ref(x); break;
        case 5: ref = gelu_tanh_ref(x); break;
        case 6: ref = 50.0 * std::tanh(x / 50.0); break;
        case 7: ref = std::isnan(x) ? x : std::min(std::max(x, -2.5), 3.5); break;
        case 8: ref = 1.0 / std::sqrt(x); break;
        case 9: ref = std::sqrt(x); break;
        case 10: ref = -x; break;
        case 11: ref = std::abs(x); break;
      }
      float reff = static_cast<float>(ref);
      // Gaudi2 TPC VPU flushes FP32 subnormals (< FLT_MIN) and x * exp_subnormal to zero.
      if (std::abs(reff) < 1e-36f && got[i] == 0.0f) continue;
      uint32_t ulp = f32_ulp_diff(got[i], reff);
      max_ulp = std::max(max_ulp, ulp);
      uint32_t tol = (op == 1 || op == 2 || op == 6) ? 3u :
                     (op == 4 || op == 3) ? 8u :
                     (op == 5) ? 16u :
                     (op == 8 || op == 9) ? 4u :
                     (op == 0 ? 4u : 0u);
      bool ok = (ulp <= tol);
      if (!ok && (op == 3 || op == 4 || op == 5) && std::isfinite(reff) && std::abs(reff) < 1e-4f) {
        double rel = std::abs(static_cast<double>(got[i]) - static_cast<double>(reff)) /
                     std::max(1e-37, std::abs(static_cast<double>(reff)));
        ok = rel < 2e-5;
      }
      if (!ok) {
        std::ostringstream oss;
        oss << "ewise_unary_f32 op=" << op << " idx=" << i << " x=" << f32_input[i]
            << " got=" << got[i] << " ref=" << reff << " ulp=" << ulp;
        throw std::runtime_error(oss.str());
      }
      if (op == 1 && f32_input[i] == 0.0f && got[i] != 0.0f) {
        throw std::runtime_error("tanh(0.0f) != 0.0f");
      }
    }
    if (op == 1) max_tanh_ulp = max_ulp;
    if (op == 2) max_sig_ulp = max_ulp;
    if (op == 4) max_gelu_erf_ulp = max_ulp;
    if (op == 5) max_gelu_tanh_ulp = max_ulp;

    // Also verify BF16 variant across all 12 ops
    std::vector<uint16_t> bf_input = in_bf16;
    if (op == 8 || op == 9) {
      for (size_t i = 0; i < bf_input.size(); ++i)
        bf_input[i] = f32_to_bf16_rne(std::abs(bf16_to_f32(bf_input[i])) + 0.0625f);
    }
    auto res_bf = run_single_node(
        ctx, "gk_ewise_unary_bf16_v1",
        {{"in", syn_type_bf16, {W, R}, bf_input.data()}},
        {{"out", syn_type_bf16, {W, R}, nullptr, (W * R + 64) * 2}},
        &p, sizeof(p), 0x5A);
    const auto& bytes_bf = res_bf.at("out");
    for (size_t b = W * R * 2; b < bytes_bf.size(); ++b) {
      if (bytes_bf[b] != 0x5A) throw std::runtime_error("ewise_unary_bf16 OOB tail write");
    }
    const uint16_t* got_bf = reinterpret_cast<const uint16_t*>(bytes_bf.data());
    for (size_t i = 0; i < W * R; ++i) {
      double x = static_cast<double>(bf16_to_f32(bf_input[i]));
      double ref = 0.0;
      switch (op) {
        case 0: ref = std::exp(x); break;
        case 1: ref = std::tanh(x); break;
        case 2: ref = 1.0 / (1.0 + std::exp(-x)); break;
        case 3: ref = (x <= -88.0) ? 0.0 : x / (1.0 + std::exp(-x)); break;
        case 4: ref = gelu_exact_ref(x); break;
        case 5: ref = gelu_tanh_ref(x); break;
        case 6: ref = 50.0 * std::tanh(x / 50.0); break;
        case 7: ref = std::isnan(x) ? x : std::min(std::max(x, -2.5), 3.5); break;
        case 8: ref = 1.0 / std::sqrt(x); break;
        case 9: ref = std::sqrt(x); break;
        case 10: ref = -x; break;
        case 11: ref = std::abs(x); break;
      }
      float reff = static_cast<float>(ref);
      if (std::abs(reff) < 1e-36f && got_bf[i] == 0) continue;
      uint16_t ref_bf = f32_to_bf16_rne(reff);
      uint32_t ulp = bf16_ulp_diff(got_bf[i], ref_bf);
      if (ulp > 1u) {
        std::ostringstream oss;
        oss << "ewise_unary_bf16 op=" << op << " idx=" << i << " x=" << x
            << " got=" << bf16_to_f32(got_bf[i]) << " ref=" << bf16_to_f32(ref_bf);
        throw std::runtime_error(oss.str());
      }
    }
  }
  std::ostringstream s;
  s << "ewise_unary(f32+bf16, W=577, ops=0..11): PASS [max_ulp: tanh=" << max_tanh_ulp
    << ", sigmoid=" << max_sig_ulp << ", gelu_erf=" << max_gelu_erf_ulp
    << ", gelu_tanh=" << max_gelu_tanh_ulp << "]";
  summary.push_back(s.str());
}

void test_gelu_and_mul_and_rms_norm(SynapseCtx& ctx, std::vector<std::string>& summary) {
  // 1. gelu_and_mul_bf16 (H=576, rows=5)
  for (int32_t approx : {0, 1}) {
    const uint64_t H = 576, R = 5;
    std::vector<uint16_t> in(2 * H * R);
    for (uint64_t r = 0; r < R; ++r) {
      for (uint64_t c = 0; c < 2 * H; ++c) {
        float v = static_cast<float>(static_cast<int>((c * 17 + r * 31) % 61) - 30) * 0.2f;
        in[r * 2 * H + c] = f32_to_bf16_rne(v);
      }
    }
    auto res = run_single_node(
        ctx, "gk_gelu_and_mul_bf16_v1",
        {{"in", syn_type_bf16, {2 * H, R}, in.data()}},
        {{"out", syn_type_bf16, {H, R}, nullptr, (H * R + 64) * 2}},
        &approx, sizeof(approx), 0x5A);
    const auto& bytes = res.at("out");
    for (size_t b = H * R * 2; b < bytes.size(); ++b) {
      if (bytes[b] != 0x5A) throw std::runtime_error("gelu_and_mul OOB write");
    }
    const uint16_t* got = reinterpret_cast<const uint16_t*>(bytes.data());
    for (uint64_t r = 0; r < R; ++r) {
      for (uint64_t c = 0; c < H; ++c) {
        double g = bf16_to_f32(in[r * 2 * H + c]);
        double u = bf16_to_f32(in[r * 2 * H + H + c]);
        double ref = (approx ? gelu_tanh_ref(g) : gelu_exact_ref(g)) * u;
        float reff = static_cast<float>(ref);
        if (std::abs(reff) < std::numeric_limits<float>::min() && got[r * H + c] == 0) continue;
        uint16_t ref_bf = f32_to_bf16_rne(reff);
        if (bf16_ulp_diff(got[r * H + c], ref_bf) > 1u) {
          throw std::runtime_error("gelu_and_mul mismatch");
        }
      }
    }
  }

  // 2. rms_norm & fused_add_rms_norm for width=64 and width=576 with weight_offset in {0.0, 1.0}
  for (uint64_t W : {64ULL, 576ULL}) {
    const uint64_t R = 5;
    for (float offset : {0.0f, 1.0f}) {
      struct { float eps; float offset; } p{1e-5f, offset};
      std::vector<uint16_t> x(W * R), res_in(W * R), gamma(W);
      for (size_t i = 0; i < W * R; ++i) {
        x[i] = f32_to_bf16_rne(static_cast<float>(static_cast<int>((i * 13) % 41) - 20) * 0.125f);
        res_in[i] = f32_to_bf16_rne(static_cast<float>(static_cast<int>((i * 7) % 29) - 14) * 0.125f);
      }
      for (size_t i = 0; i < W; ++i) {
        gamma[i] = f32_to_bf16_rne(0.5f + static_cast<float>(i % 9) * 0.1f);
      }
      // Unfused rms_norm
      auto out1 = run_single_node(
          ctx, "gk_rms_norm_bf16_v1",
          {{"x", syn_type_bf16, {W, R}, x.data()},
           {"gamma", syn_type_bf16, {W}, gamma.data()}},
          {{"out", syn_type_bf16, {W, R}, nullptr, (W * R + 64) * 2}},
          &p, sizeof(p), 0x5A);
      const auto& b1 = out1.at("out");
      for (size_t b = W * R * 2; b < b1.size(); ++b) {
        if (b1[b] != 0x5A) throw std::runtime_error("rms_norm OOB tail write");
      }
      const uint16_t* got1 = reinterpret_cast<const uint16_t*>(b1.data());
      for (uint64_t r = 0; r < R; ++r) {
        double ss = 0.0;
        for (uint64_t c = 0; c < W; ++c) {
          double v = bf16_to_f32(x[r * W + c]);
          ss += v * v;
        }
        double rstd = 1.0 / std::sqrt(ss / static_cast<double>(W) + p.eps);
        for (uint64_t c = 0; c < W; ++c) {
          double g = static_cast<double>(bf16_to_f32(gamma[c])) + static_cast<double>(offset);
          double ref = static_cast<double>(bf16_to_f32(x[r * W + c])) * rstd * g;
          if (bf16_ulp_diff(got1[r * W + c], f32_to_bf16_rne(static_cast<float>(ref))) > 2u) {
            throw std::runtime_error("rms_norm mismatch");
          }
        }
      }

      // Fused add + rms_norm
      auto out2 = run_single_node(
          ctx, "gk_fused_add_rms_norm_bf16_v1",
          {{"x", syn_type_bf16, {W, R}, x.data()},
           {"res", syn_type_bf16, {W, R}, res_in.data()},
           {"gamma", syn_type_bf16, {W}, gamma.data()}},
          {{"res_out", syn_type_bf16, {W, R}, nullptr, (W * R + 64) * 2},
           {"norm_out", syn_type_bf16, {W, R}, nullptr, (W * R + 64) * 2}},
          &p, sizeof(p), 0x5A);
      const auto& b2r = out2.at("res_out");
      const auto& b2n = out2.at("norm_out");
      for (size_t b = W * R * 2; b < b2n.size(); ++b) {
        if (b2n[b] != 0x5A || b2r[b] != 0x5A) throw std::runtime_error("fused_rms_norm OOB tail write");
      }
      const uint16_t* got_r = reinterpret_cast<const uint16_t*>(b2r.data());
      const uint16_t* got_n = reinterpret_cast<const uint16_t*>(b2n.data());
      for (uint64_t r = 0; r < R; ++r) {
        double ss = 0.0;
        std::vector<float> summed(W);
        for (uint64_t c = 0; c < W; ++c) {
          float s = bf16_to_f32(x[r * W + c]) + bf16_to_f32(res_in[r * W + c]);
          uint16_t s_bf = f32_to_bf16_rne(s);
          if (bf16_ulp_diff(got_r[r * W + c], s_bf) > 0u) {
            throw std::runtime_error("fused_add_rms_norm residual mismatch");
          }
          summed[c] = bf16_to_f32(s_bf);
          ss += static_cast<double>(summed[c]) * summed[c];
        }
        double rstd = 1.0 / std::sqrt(ss / static_cast<double>(W) + p.eps);
        for (uint64_t c = 0; c < W; ++c) {
          double g = static_cast<double>(bf16_to_f32(gamma[c])) + static_cast<double>(offset);
          double ref = static_cast<double>(summed[c]) * rstd * g;
          if (bf16_ulp_diff(got_n[r * W + c], f32_to_bf16_rne(static_cast<float>(ref))) > 2u) {
            throw std::runtime_error("fused_add_rms_norm norm mismatch");
          }
        }
      }
    }
  }
  summary.push_back("gelu_and_mul(H=576, approx=0/1) & rms_norm/fused_add_rms_norm(W=64,576, offset=0,1): PASS");
}

void test_rotary_pos(SynapseCtx& ctx, std::vector<std::string>& summary) {
  const uint64_t QH = 6, KH = 2, HD = 128, RD = 64, R = 7, CACHE_R = 32;
  std::vector<uint16_t> q(QH * HD * R), k(KH * HD * R), cache(RD * CACHE_R);
  std::vector<int32_t> pos{-5, 0, 1, 13, 31, 9999, 7};  // tests clamping [0, 31]
  for (size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16_rne(static_cast<float>((int(i % 29) - 14)) * 0.25f);
  for (size_t i = 0; i < k.size(); ++i) k[i] = f32_to_bf16_rne(static_cast<float>((int(i % 23) - 11)) * 0.25f);
  for (uint64_t r = 0; r < CACHE_R; ++r) {
    for (uint64_t d = 0; d < RD / 2; ++d) {
      double angle = static_cast<double>(r) * (0.1 + 0.03 * d);
      cache[r * RD + d] = f32_to_bf16_rne(static_cast<float>(std::cos(angle)));
      cache[r * RD + RD / 2 + d] = f32_to_bf16_rne(static_cast<float>(std::sin(angle)));
    }
  }

  for (int32_t is_neox : {0, 1}) {
    struct { int32_t hd; int32_t rd; int32_t neox; } rp{int32_t(HD), int32_t(RD), is_neox};
    auto res = run_single_node(
        ctx, "gk_rotary_pos_bf16_v1",
        {{"q", syn_type_bf16, {QH * HD, R}, q.data()},
         {"k", syn_type_bf16, {KH * HD, R}, k.data()},
         {"pos", syn_type_int32, {R}, pos.data()},
         {"cache", syn_type_bf16, {RD, CACHE_R}, cache.data()}},
        {{"qo", syn_type_bf16, {QH * HD, R}},
         {"ko", syn_type_bf16, {KH * HD, R}}},
        &rp, sizeof(rp));
    auto check = [&](const std::vector<uint16_t>& src, const uint16_t* got, uint64_t heads) {
      for (uint64_t r = 0; r < R; ++r) {
        int32_t p = std::max<int32_t>(0, std::min<int32_t>(pos[r], static_cast<int32_t>(CACHE_R) - 1));
        for (uint64_t h = 0; h < heads; ++h) {
          uint64_t base = r * heads * HD + h * HD;
          if (is_neox) {
            for (uint64_t d = 0; d < RD / 2; ++d) {
              float x0 = bf16_to_f32(src[base + d]);
              float x1 = bf16_to_f32(src[base + RD / 2 + d]);
              float c = bf16_to_f32(cache[p * RD + d]);
              float s = bf16_to_f32(cache[p * RD + RD / 2 + d]);
              uint16_t y0 = f32_to_bf16_rne(x0 * c - x1 * s);
              uint16_t y1 = f32_to_bf16_rne(x1 * c + x0 * s);
              if (bf16_ulp_diff(got[base + d], y0) > 1u ||
                  bf16_ulp_diff(got[base + RD / 2 + d], y1) > 1u)
                throw std::runtime_error("rotary_pos neox mismatch");
            }
          } else {
            for (uint64_t d = 0; d < RD / 2; ++d) {
              float x0 = bf16_to_f32(src[base + 2 * d]);
              float x1 = bf16_to_f32(src[base + 2 * d + 1]);
              float c = bf16_to_f32(cache[p * RD + d]);
              float s = bf16_to_f32(cache[p * RD + RD / 2 + d]);
              uint16_t y0 = f32_to_bf16_rne(x0 * c - x1 * s);
              uint16_t y1 = f32_to_bf16_rne(x1 * c + x0 * s);
              if (bf16_ulp_diff(got[base + 2 * d], y0) > 1u ||
                  bf16_ulp_diff(got[base + 2 * d + 1], y1) > 1u)
                throw std::runtime_error("rotary_pos interleaved mismatch");
            }
          }
          for (uint64_t d = RD; d < HD; ++d) {
            if (got[base + d] != src[base + d]) {
              throw std::runtime_error("rotary_pos passthrough mismatch");
            }
          }
        }
      }
    };
    check(q, reinterpret_cast<const uint16_t*>(res.at("qo").data()), QH);
    check(k, reinterpret_cast<const uint16_t*>(res.at("ko").data()), KH);
  }
  summary.push_back("rotary_pos_bf16(QH=6,KH=2,HD=128,RD=64,is_neox=0/1,pos_clamping): PASS");
}

void test_kv_and_attn_extensions(SynapseCtx& ctx, std::vector<std::string>& summary) {
  // 1. kv_store_head_major (d=64 with heads=3 -> width=192) & d=256 (heads=2 -> width=512)
  for (auto [guid, d, heads] : {std::tuple{"gk_kv_store_head_major_bf16_v1", 64ULL, 3ULL},
                                std::tuple{"gk_kv_store_head_major_bf16_v1", 256ULL, 2ULL}}) {
    const uint64_t rows = 4, valid = 3, slots = 32;
    std::vector<uint16_t> key(heads * d * rows), val(heads * d * rows), pool(d * 2 * heads * slots, 0x3C00);
    std::vector<int32_t> locs{5, 17, 29, 3};  // row 3 >= valid should NOT be written
    std::vector<int32_t> meta{static_cast<int32_t>(valid), static_cast<int32_t>(slots), static_cast<int32_t>(heads)};
    for (size_t i = 0; i < key.size(); ++i) {
      key[i] = f32_to_bf16_rne(static_cast<float>(i % 97) * 0.5f + 1.0f);
      val[i] = f32_to_bf16_rne(-static_cast<float>(i % 89) * 0.5f - 1.0f);
    }
    auto res = run_single_node(
        ctx, guid,
        {{"key", syn_type_bf16, {heads * d, rows}, key.data()},
         {"val", syn_type_bf16, {heads * d, rows}, val.data()},
         {"locs", syn_type_int32, {rows}, locs.data()},
         {"meta", syn_type_int32, {3}, meta.data()},
         {"old_pool", syn_type_bf16, {d, 2 * heads * slots}, pool.data()}},
        {{"pool", syn_type_bf16, {d, 2 * heads * slots}, nullptr, 0, "old_pool"}},
        nullptr, 0);
    const uint16_t* got = reinterpret_cast<const uint16_t*>(res.at("pool").data());
    for (uint64_t r = 0; r < valid; ++r) {
      uint64_t slot = static_cast<uint64_t>(locs[r]);
      for (uint64_t h = 0; h < heads; ++h) {
        for (uint64_t c = 0; c < d; ++c) {
          uint64_t src_i = r * (heads * d) + h * d + c;
          uint64_t k_dst = (h * slots + slot) * d + c;
          uint64_t v_dst = ((heads + h) * slots + slot) * d + c;
          if (got[k_dst] != key[src_i] || got[v_dst] != val[src_i]) {
            throw std::runtime_error(std::string(guid) + " store mismatch");
          }
        }
      }
    }
    // Check untouched slot 3 (row 3 >= valid)
    for (uint64_t c = 0; c < d; ++c) {
      if (got[3 * d + c] != 0x3C00) throw std::runtime_error("kv_store wrote padded row!");
    }
  }

  // 2. attn_decode_normalize_bf16 keys=64 (OOB guard) & attn_online_merge/finalize width=256
  {
    const uint64_t K = 64, H = 4;
    std::vector<float> exp_v(K * H), sums(64 * H);
    for (uint64_t h = 0; h < H; ++h) {
      float s = 0.0f;
      for (uint64_t k = 0; k < K; ++k) {
        exp_v[h * K + k] = static_cast<float>(k + 1) * 0.1f + h;
        s += exp_v[h * K + k];
      }
      for (uint64_t i = 0; i < 64; ++i) sums[h * 64 + i] = s;
    }
    auto res = run_single_node(
        ctx, "gk_attn_decode_normalize_bf16_v1",
        {{"exp", syn_type_single, {K, 1, H}, exp_v.data()},
         {"sums", syn_type_single, {64, H}, sums.data()}},
        {{"out", syn_type_bf16, {K, 1, H}, nullptr, (K * H + 64) * 2}},
        nullptr, 0, 0x5A);
    const auto& bytes = res.at("out");
    for (size_t b = K * H * 2; b < bytes.size(); ++b) {
      if (bytes[b] != 0x5A) throw std::runtime_error("decode_normalize(K=64) OOB tail write");
    }
    const uint16_t* got = reinterpret_cast<const uint16_t*>(bytes.data());
    for (uint64_t h = 0; h < H; ++h) {
      for (uint64_t k = 0; k < K; ++k) {
        uint16_t ref = f32_to_bf16_rne(exp_v[h * K + k] / sums[h * 64]);
        if (bf16_ulp_diff(got[h * K + k], ref) > 1u) {
          throw std::runtime_error("decode_normalize(K=64) value mismatch");
        }
      }
    }
  }
  {
    const uint64_t W = 256, R = 67, H = 3, SR = 128;
    std::vector<float> pm(SR * H, 1.0f), ps(SR * H, 2.0f), pd(W * R * H);
    std::vector<float> tm(SR * H, 1.5f), ts(SR * H, 3.0f), td(W * R * H);
    for (size_t i = 0; i < pd.size(); ++i) {
      pd[i] = static_cast<float>((i % 19) + 1) * 0.1f;
      td[i] = static_cast<float>((i % 23) + 1) * 0.1f;
    }
    int32_t prev_unscaled = 1;
    auto mres = run_single_node(
        ctx, "gk_attn_online_merge_f32_v1",
        {{"pm", syn_type_single, {SR, H}, pm.data()},
         {"ps", syn_type_single, {SR, H}, ps.data()},
         {"pd", syn_type_single, {W, R, H}, pd.data()},
         {"tm", syn_type_single, {SR, H}, tm.data()},
         {"ts", syn_type_single, {SR, H}, ts.data()},
         {"td", syn_type_single, {W, R, H}, td.data()}},
        {{"om", syn_type_single, {SR, H}},
         {"os", syn_type_single, {SR, H}},
         {"od", syn_type_single, {W, R, H}}},
        &prev_unscaled, sizeof(prev_unscaled));
    const float* om = reinterpret_cast<const float*>(mres.at("om").data());
    const float* os = reinterpret_cast<const float*>(mres.at("os").data());
    const float* od = reinterpret_cast<const float*>(mres.at("od").data());
    // Finalize width=256
    auto fres = run_single_node(
        ctx, "gk_attn_online_finalize_bf16_v1",
        {{"os", syn_type_single, {SR, H}, os},
         {"od", syn_type_single, {W, R, H}, od}},
        {{"out", syn_type_bf16, {W, R, H}}},
        nullptr, 0);
    const uint16_t* got = reinterpret_cast<const uint16_t*>(fres.at("out").data());
    float e_diff = std::exp(1.0f - 1.5f);
    float expected_sum = 2.0f * e_diff + 3.0f;
    for (uint64_t h = 0; h < H; ++h) {
      for (uint64_t r = 0; r < R; ++r) {
        if (std::abs(om[h * SR + r] - 1.5f) > 1e-5f || std::abs(os[h * SR + r] - expected_sum) > 1e-4f)
          throw std::runtime_error("online_merge(W=256) stats mismatch");
        for (uint64_t c = 0; c < W; ++c) {
          size_t idx = (h * R + r) * W + c;
          float ref = (2.0f * e_diff * pd[idx] + 3.0f * td[idx]) / expected_sum;
          if (bf16_ulp_diff(got[idx], f32_to_bf16_rne(ref)) > 1u) {
            throw std::runtime_error("online_finalize(W=256) mismatch");
          }
        }
      }
    }
  }

  // 3. SWA + Softcap on tile_softmax, direct_softmax, and decode_exp + fully-masked row check!
  {
    const uint64_t K = 128, R = 8, H = 2, SR = 64;
    std::vector<float> scores(K * R * H);
    for (size_t i = 0; i < scores.size(); ++i)
      scores[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.4f;
    // key_start = 10, valid_keys = 64 -> keys represent global positions [10 .. 73]
    // pos[0] = 5 -> ALL keys >= 10 > 5 -> fully masked causal row!
    // pos[1] = 80, swa = 4 -> window [77..80] -> ALL keys <= 73 < 77 -> fully masked SWA row!
    // pos[2] = 20, swa = 4 -> window [17..20], keys index [7..10] unmasked!
    std::vector<int32_t> pos{5, 80, 20, 30, 45, 73, 10, 12};
    std::vector<int32_t> hdr{static_cast<int32_t>(R), 64, 10};
    struct { float scale; float softcap; int32_t swa; } sm{0.25f, 50.0f, 4};

    auto tres = run_single_node(
        ctx, "gk_attn_tile_softmax_bf16_v1",
        {{"sc", syn_type_single, {K, R, H}, scores.data()},
         {"pos", syn_type_int32, {R}, pos.data()},
         {"hdr", syn_type_int32, {3}, hdr.data()}},
        {{"pr", syn_type_bf16, {K, R, H}},
         {"tmax", syn_type_single, {SR, H}},
         {"tsum", syn_type_single, {SR, H}}},
        &sm, sizeof(sm));
    auto dres = run_single_node(
        ctx, "gk_attn_direct_softmax_bf16_v1",
        {{"sc", syn_type_single, {K, R, H}, scores.data()},
         {"pos", syn_type_int32, {R}, pos.data()},
         {"hdr", syn_type_int32, {3}, hdr.data()}},
        {{"pr", syn_type_bf16, {K, R, H}}},
        &sm, sizeof(sm));

    const uint16_t* tpr = reinterpret_cast<const uint16_t*>(tres.at("pr").data());
    const float* tmax = reinterpret_cast<const float*>(tres.at("tmax").data());
    const float* tsum = reinterpret_cast<const float*>(tres.at("tsum").data());
    const uint16_t* dpr = reinterpret_cast<const uint16_t*>(dres.at("pr").data());

    for (uint64_t h = 0; h < H; ++h) {
      for (uint64_t r = 0; r < R; ++r) {
        int32_t lo = std::max(0, pos[r] - sm.swa + 1 - hdr[2]);
        int32_t hi = std::min(hdr[1] - 1, pos[r] - hdr[2]);
        double row_max = -std::numeric_limits<double>::infinity();
        std::vector<double> capped(K, 0.0);
        for (int32_t k = lo; k <= hi; ++k) {
          if (k >= 0 && k < hdr[1]) {
            double v = 50.0 * std::tanh(scores[(h * R + r) * K + k] * (0.25 / 50.0));
            capped[k] = v;
            row_max = std::max(row_max, v);
          }
        }
        double row_sum = 0.0;
        if (std::isfinite(row_max)) {
          for (int32_t k = std::max(0, lo); k <= std::min(hdr[1] - 1, hi); ++k) {
            row_sum += std::exp(capped[k] - row_max);
          }
          if (std::abs(tmax[h * SR + r] - row_max) > 1e-4 || std::abs(tsum[h * SR + r] - row_sum) > 1e-4)
            throw std::runtime_error("tile_softmax SWA+softcap stats mismatch");
        } else {
          // Fully masked row must have max=-inf, sum=0, ZERO NaNs!
          if (!std::isinf(tmax[h * SR + r]) || tmax[h * SR + r] > 0 || tsum[h * SR + r] != 0.0f)
            throw std::runtime_error("tile_softmax fully-masked row stats failed");
        }
        for (uint64_t k = 0; k < K; ++k) {
          uint64_t idx = (h * R + r) * K + k;
          float tp = bf16_to_f32(tpr[idx]), dp = bf16_to_f32(dpr[idx]);
          if (std::isnan(tp) || std::isnan(dp))
            throw std::runtime_error("softmax SWA+softcap produced NaN!");
          double exp_v = (row_sum > 0.0 && static_cast<int32_t>(k) >= lo && static_cast<int32_t>(k) <= hi)
                             ? std::exp(capped[k] - row_max) / row_sum
                             : 0.0;
          uint16_t ref_bf = f32_to_bf16_rne(static_cast<float>(exp_v));
          if (bf16_ulp_diff(tpr[idx], ref_bf) > 1u || bf16_ulp_diff(dpr[idx], ref_bf) > 1u) {
            throw std::runtime_error("softmax SWA+softcap prob mismatch");
          }
        }
      }
    }

    // Long decode exp + normalize (keys=1024, SWA=64, softcap=30.0)
    const uint64_t DK = 1024, DH = 2;
    std::vector<float> dscores(DK * DH);
    for (size_t i = 0; i < dscores.size(); ++i)
      dscores[i] = static_cast<float>(static_cast<int>(i % 43) - 21) * 0.3f;
    std::vector<int32_t> dpos{700}, dhdr{1, 800, 0};
    struct { float scale; float softcap; int32_t swa; } dsm{0.125f, 30.0f, 64};
    auto dexp = run_single_node(
        ctx, "gk_attn_decode_exp_f32_v1",
        {{"sc", syn_type_single, {DK, 1, DH}, dscores.data()},
         {"pos", syn_type_int32, {1}, dpos.data()},
         {"hdr", syn_type_int32, {3}, dhdr.data()}},
        {{"exp", syn_type_single, {DK, 1, DH}},
         {"sums", syn_type_single, {64, DH}}},
        &dsm, sizeof(dsm));
    auto dnorm = run_single_node(
        ctx, "gk_attn_decode_normalize_bf16_v1",
        {{"exp", syn_type_single, {DK, 1, DH}, dexp.at("exp").data()},
         {"sums", syn_type_single, {64, DH}, dexp.at("sums").data()}},
        {{"probs", syn_type_bf16, {DK, 1, DH}}},
        nullptr, 0);
    const uint16_t* got_dp = reinterpret_cast<const uint16_t*>(dnorm.at("probs").data());
    for (uint64_t h = 0; h < DH; ++h) {
      int32_t lo = dpos[0] - dsm.swa + 1;
      int32_t hi = dpos[0];
      double rmax = -1e30;
      std::vector<double> cap(DK, 0.0);
      for (int32_t k = lo; k <= hi; ++k) {
        cap[k] = 30.0 * std::tanh(dscores[h * DK + k] * (0.125 / 30.0));
        rmax = std::max(rmax, cap[k]);
      }
      double rsum = 0.0;
      for (int32_t k = lo; k <= hi; ++k) rsum += std::exp(cap[k] - rmax);
      for (uint64_t k = 0; k < DK; ++k) {
        double p = (static_cast<int32_t>(k) >= lo && static_cast<int32_t>(k) <= hi)
                       ? std::exp(cap[k] - rmax) / rsum
                       : 0.0;
        if (bf16_ulp_diff(got_dp[h * DK + k], f32_to_bf16_rne(static_cast<float>(p))) > 1u)
          throw std::runtime_error("decode_exp+norm SWA+softcap mismatch");
      }
    }
  }
  summary.push_back("kv_store_head_major(d=64/192,d=256) + online(W=256) + decode_norm(K=64) + SWA+Softcap: PASS");
}

void test_paged_free(SynapseCtx& ctx, std::vector<std::string>& summary) {
  const int32_t page_size = 32;  // non-16/64!
  const int32_t max_pages = 16000;
  const int32_t bitset_words = (max_pages + 31) / 32 + 1;
  const uint64_t N = 8;
  std::vector<int32_t> indices{
      12000 * 32 + 5, 0,
      12000 * 32 + 19, 0,
      1 * 32 + 0, 0,
      15999 * 32 + 31, 0,
      1 * 32 + 7, 0,
      12000 * 32 + 0, 0,
      0, 0, 0, 0};
  std::vector<int32_t> meta{6, max_pages, bitset_words};
  std::vector<int32_t> bitset(bitset_words, 0);
  auto res = run_single_node(
      ctx, "gk_paged_free_i32_v1",
      {{"idx", syn_type_int32, {2, N}, indices.data()},
       {"meta", syn_type_int32, {3}, meta.data()},
       {"bitset", syn_type_int32, {static_cast<uint64_t>(bitset_words)}, bitset.data()}},
      {{"pages", syn_type_int32, {2, N}},
       {"cnt", syn_type_int32, {1}}},
      &page_size, sizeof(page_size));
  const int32_t cnt = *reinterpret_cast<const int32_t*>(res.at("cnt").data());
  const int32_t* pg = reinterpret_cast<const int32_t*>(res.at("pages").data());
  if (cnt != 3 || pg[0] != 12000 || pg[2] != 1 || pg[4] != 15999) {
    throw std::runtime_error("paged_free deduplication >8192 failed");
  }

  // Invalid page > max_pages must fail cleanly with result_count = -1
  std::vector<int32_t> bad_idx = indices;
  bad_idx[2] = 16001 * 32;
  auto res_bad = run_single_node(
      ctx, "gk_paged_free_i32_v1",
      {{"idx", syn_type_int32, {2, N}, bad_idx.data()},
       {"meta", syn_type_int32, {3}, meta.data()},
       {"bitset", syn_type_int32, {static_cast<uint64_t>(bitset_words)}, bitset.data()}},
      {{"pages", syn_type_int32, {2, N}},
       {"cnt", syn_type_int32, {1}}},
      &page_size, sizeof(page_size));
  if (*reinterpret_cast<const int32_t*>(res_bad.at("cnt").data()) != -1) {
    throw std::runtime_error("paged_free failed to reject OOB page");
  }
  summary.push_back("paged_free_i32(page_size=32, page=12000/15999 >8192, dedup+OOB): PASS");
}

void test_sample_vocab(SynapseCtx& ctx, std::vector<std::string>& summary) {
  const int32_t V = 5000;
  const uint64_t DIM0 = 5120, R = 32;
  std::vector<float> logits_f32(DIM0 * R, -100.0f);
  std::vector<float> temps(R, 1.0f), top_ps(R, 1.0f), min_ps(R, 0.0f);
  std::vector<int32_t> top_ks(R, 0), seeds(2 * R, 0);

  auto set_row_seed = [&](uint64_t r, uint32_t s0, uint32_t s1) {
    seeds[r * 2] = static_cast<int32_t>(s0);
    seeds[r * 2 + 1] = static_cast<int32_t>(s1);
  };

  // Row 0: greedy via temp=0 with NaN at index 5 and finite max at index 4321
  logits_f32[0 * DIM0 + 5] = std::numeric_limits<float>::quiet_NaN();
  logits_f32[0 * DIM0 + 4321] = 7.5f;
  temps[0] = 0.0f;

  // Row 1: greedy via top_k=1 with higher padded tail value at index 5050 (>= V!) ignored
  logits_f32[1 * DIM0 + 5050] = 999.0f;
  logits_f32[1 * DIM0 + 17] = 3.0f;
  top_ks[1] = 1;

  // Row 2: exact tie Top-K quota! Indices 0..399 all have logit 5.0, top_k=200, top_p=1, temp=1
  // Only indices in [0..199] may be sampled; 200..399 MUST NEVER be sampled!
  for (int i = 0; i < 400; ++i) logits_f32[2 * DIM0 + i] = 5.0f;
  top_ks[2] = 200;
  set_row_seed(2, 12345u, 67890u);

  // Row 3: min_p=0.5 with max at idx 10 (logit 0.0, weight=1.0), idx 20 (logit=-0.5 -> w=0.606 >=0.5),
  // idx 30 (logit=-2.0 -> w=0.135 <0.5), temp=1. Only {10, 20} allowed!
  logits_f32[3 * DIM0 + 10] = 0.0f;
  logits_f32[3 * DIM0 + 20] = -0.5f;
  logits_f32[3 * DIM0 + 30] = -2.0f;
  min_ps[3] = 0.5f;
  set_row_seed(3, 111u, 222u);

  // Row 4: top_p=0.6 with dominant idx 88 (prob ~0.73), idx 89 (prob ~0.27) -> only idx 88 kept!
  logits_f32[4 * DIM0 + 88] = 2.0f;
  logits_f32[4 * DIM0 + 89] = 1.0f;
  top_ps[4] = 0.6f;
  set_row_seed(4, 99999u, 88888u);

  // Rows 5 & 6: identical logits (idx 100..103 = 2.0) and IDENTICAL Philox seed -> exact match!
  // Rows 7..31: identical logits (idx 200: ~0.70, idx 201: ~0.25, idx 202: ~0.05) across 25 distinct seeds!
  for (uint64_t r = 5; r <= 6; ++r) {
    for (int i = 100; i < 120; ++i) logits_f32[r * DIM0 + i] = 2.0f;
    set_row_seed(r, 424242u, 131313u);
  }
  for (uint64_t r = 7; r < R; ++r) {
    logits_f32[r * DIM0 + 200] = std::log(70.0f);
    logits_f32[r * DIM0 + 201] = std::log(25.0f);
    logits_f32[r * DIM0 + 202] = std::log(5.0f);
    set_row_seed(r, static_cast<uint32_t>(1000 + r * 77), static_cast<uint32_t>(2000 + r * 91));
  }

  struct { int32_t vocab; int32_t nan_mask; } sp{V, 1};
  auto res_f32 = run_single_node(
      ctx, "gk_sample_vocab_f32_v1",
      {{"logits", syn_type_single, {DIM0, R}, logits_f32.data()},
       {"temps", syn_type_single, {R}, temps.data()},
       {"top_ps", syn_type_single, {R}, top_ps.data()},
       {"top_ks", syn_type_int32, {R}, top_ks.data()},
       {"min_ps", syn_type_single, {R}, min_ps.data()},
       {"seeds", syn_type_int32, {2, R}, seeds.data()}},
      {{"out", syn_type_int32, {R}}},
      &sp, sizeof(sp));
  const int32_t* ids = reinterpret_cast<const int32_t*>(res_f32.at("out").data());
  if (ids[0] != 4321) throw std::runtime_error("sample_vocab NaN mask + greedy failed");
  if (ids[1] != 17) throw std::runtime_error("sample_vocab padded dim0 vocab_size guard or top_k=1 failed");
  if (ids[2] < 0 || ids[2] >= 200) throw std::runtime_error("sample_vocab tie Top-K quota failed");
  if (ids[3] != 10 && ids[3] != 20) throw std::runtime_error("sample_vocab min_p filter failed");
  if (ids[4] != 88) throw std::runtime_error("sample_vocab top_p=0.6 failed");
  if (ids[5] != ids[6] || ids[5] < 100 || ids[5] >= 120)
    throw std::runtime_error("sample_vocab deterministic Philox seed failed");
  int c200 = 0;
  for (uint64_t r = 7; r < R; ++r) {
    if (ids[r] < 200 || ids[r] > 202) throw std::runtime_error("sample_vocab multi-row leak");
    if (ids[r] == 200) ++c200;
  }
  if (c200 < 10) throw std::runtime_error("sample_vocab CDF distribution skew anomalous");

  // Verify BF16 variant + tie quota across multiple seeds
  std::vector<uint16_t> logits_bf16(DIM0 * R);
  for (size_t i = 0; i < logits_f32.size(); ++i) logits_bf16[i] = f32_to_bf16_rne(logits_f32[i]);
  auto res_bf16 = run_single_node(
      ctx, "gk_sample_vocab_bf16_v1",
      {{"logits", syn_type_bf16, {DIM0, R}, logits_bf16.data()},
       {"temps", syn_type_single, {R}, temps.data()},
       {"top_ps", syn_type_single, {R}, top_ps.data()},
       {"top_ks", syn_type_int32, {R}, top_ks.data()},
       {"min_ps", syn_type_single, {R}, min_ps.data()},
       {"seeds", syn_type_int32, {2, R}, seeds.data()}},
      {{"out", syn_type_int32, {R}}},
      &sp, sizeof(sp));
  const int32_t* ids_bf = reinterpret_cast<const int32_t*>(res_bf16.at("out").data());
  if (ids_bf[0] != 4321 || ids_bf[1] != 17 || ids_bf[2] < 0 || ids_bf[2] >= 200 ||
      (ids_bf[3] != 10 && ids_bf[3] != 20) || ids_bf[4] != 88 || ids_bf[5] != ids_bf[6]) {
    throw std::runtime_error("sample_vocab_bf16 check failed");
  }
  summary.push_back("sample_vocab(f32+bf16, V=5000/5120, R=32, greedy/NaN/tie-TopK/min_p/top_p/Philox): PASS");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string json_out;
    for (int i = 1; i + 1 < argc; ++i) {
      if (std::strcmp(argv[i], "--json") == 0) json_out = argv[i + 1];
    }

    SynapseCtx ctx;
    std::vector<std::string> summary;
    test_ewise_and_math(ctx, summary);
    test_gelu_and_mul_and_rms_norm(ctx, summary);
    test_rotary_pos(ctx, summary);
    test_kv_and_attn_extensions(ctx, summary);
    test_paged_free(ctx, summary);
    test_sample_vocab(ctx, summary);

    for (const auto& s : summary) std::cout << s << '\n';
    if (!json_out.empty()) {
      std::ofstream f(json_out);
      f << "{\n  \"status\": \"PASS\",\n  \"visible_modules\": \"0\",\n  \"suites\": [\n";
      for (size_t i = 0; i < summary.size(); ++i) {
        f << "    \"" << summary[i] << "\"" << (i + 1 < summary.size() ? "," : "") << "\n";
      }
      f << "  ]\n}\n";
    }
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "FAIL: " << err.what() << '\n';
    return 1;
  }
}
