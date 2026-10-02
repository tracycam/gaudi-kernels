// CPU-only correctly-rounded FP32 FMA array primitive. No device/runtime APIs.
// Build without fast-math; std::fma is explicit and must not become mul+add.
#include <cfenv>
#include <cmath>
#include <cstdint>

extern "C" int gk_fp32_oracle_version() { return 1; }
extern "C" int gk_fp32_oracle_fma(const float* a, const float* b,
                                 const float* c, float* out, uint64_t count) {
  if (!a || !b || !c || !out || std::fegetround() != FE_TONEAREST) return -1;
  for (uint64_t i = 0; i < count; ++i) out[i] = std::fma(a[i], b[i], c[i]);
  return 0;
}
