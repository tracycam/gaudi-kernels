// FP32 Neumaier reduction with its residual preserved at BF16 midpoints.
// A rounded FP32 sum can land exactly on a BF16 midpoint even when the
// compensated value lies to one side; use the residual to choose that side.
// This model QKV kernel assumes nonzero compensation terms are FP32 normal.
// It does not implement an arbitrary-subnormal-input reduction contract.
static inline float64 abs32(float64 value) {
  return (float64)((uint64)value & 0x7fffffff);
}

static inline float64 round_input(float64 sum, float64 correction, float64 bias) {
  float64 first = sum + correction;
  float64 first_error = v_f32_sel_geq_f32_b(
      abs32(sum), abs32(correction), (sum - first) + correction,
      (correction - first) + sum);
  float64 total = first + bias;
  float64 bias_error = v_f32_sel_geq_f32_b(
      abs32(first), abs32(bias), (first - total) + bias,
      (bias - total) + first);
  float64 error = first_error + bias_error;
  uint64 bits = (uint64)total;
  uint64 toward_positive = v_u32_sel_eq_u32_b(bits & 0x80000000, 0, bits + 1, bits - 1);
  uint64 toward_negative = v_u32_sel_eq_u32_b(bits & 0x80000000, 0, bits - 1, bits + 1);
  float64 adjusted = v_f32_sel_grt_f32_b(error, 0.f, (float64)toward_positive, total);
  adjusted = v_f32_sel_less_f32_b(error, 0.f, (float64)toward_negative, adjusted);
  return v_f32_sel_eq_u32_b(bits & 0xffff, 0x8000, adjusted, total);
}

void main(tensor partial, tensor activation_scales, tensor weight_scales,
          tensor bias, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(partial, 2);
  for (int row = begin[1]; row < end[1]; ++row)
    for (int nb = begin[0]; nb < end[0]; ++nb) {
      int col = nb * 128;
      float64 sumlo = 0, sumhi = 0, corrlo = 0, corrhi = 0;
      for (int g = 0; g < groups; ++g) {
        int5 wp = {g, nb, 0, 0, 0}, ap = {0, row, g, 0, 0};
        int5 pp = {col, row, g, 0, 0};
        float64 ws = v_f32_ld_g(gen_addr(wp, weight_scales));
        float64 as = v_f32_ld_g(gen_addr(ap, activation_scales));
        float64 factor = ws * as;
        float64 lo = v_f32_ld_tnsr_b(pp, partial);
        pp[0] += 64;
        float64 hi = v_f32_ld_tnsr_b(pp, partial);
#define ACCUM(S, V) { \
        float64 term = V * factor; \
        float64 product_error = v_f32_mac_b(V, factor, -term); \
        float64 updated = sum##S + term; \
        float64 first = (sum##S - updated) + term; \
        float64 second = (term - updated) + sum##S; \
        float64 residual = v_f32_sel_geq_f32_b(abs32(sum##S), abs32(term), first, second); \
        corr##S = corr##S + (residual + product_error); \
        sum##S = updated; \
      }
        ACCUM(lo, lo)
        ACCUM(hi, hi)
#undef ACCUM
      }
      int5 b = {col, 0, 0, 0, 0};
      float64 blo = v_f32_ld_tnsr_b(b, bias);
      b[0] += 64;
      float64 bhi = v_f32_ld_tnsr_b(b, bias);
      float128 pair = {round_input(sumlo, corrlo, blo), round_input(sumhi, corrhi, bhi)};
      int5 dst = {col, row, 0, 0, 0};
      v_bf16_st_tnsr(dst, output, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
    }
}
