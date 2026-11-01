// Compact original block scales, broadcast directly into vector registers.
// The compiler must keep [G,M,N] partials in SRAM; audit separately.
#ifndef ROWS
#define ROWS 1
#endif
#ifndef MAX_1ROW
#define MAX_1ROW 4
#endif
#ifndef MAX_2ROW
#define MAX_2ROW 16
#endif

static inline void reduce_1row(
    tensor partial, tensor activation_scales, tensor weight_scales, tensor output,
    int col, int nb, int row, int groups, float64 blo, float64 bhi) {
  float64 lo0 = 0, hi0 = 0;
  int5 wp = {0, nb, 0, 0, 0}, ap = {0, row, 0, 0, 0};
  int5 plo = {col, row, 0, 0, 0}, phi = {col + 64, row, 0, 0, 0};
  int g4 = groups & ~3, g = 0;
  for (; g < g4; g += 4) {
    int g1 = g | 1, g2 = g | 2, g3 = g | 3;
    wp[0] = g; ap[2] = g; plo[2] = g; phi[2] = g;
    float64 ws0 = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as0 = v_f32_ld_g(gen_addr(ap, activation_scales));
    float64 xlo0 = v_f32_ld_tnsr_b(plo, partial), xhi0 = v_f32_ld_tnsr_b(phi, partial);

    wp[0] = g1; ap[2] = g1; plo[2] = g1; phi[2] = g1;
    float64 ws1 = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as1 = v_f32_ld_g(gen_addr(ap, activation_scales));
    float64 xlo1 = v_f32_ld_tnsr_b(plo, partial), xhi1 = v_f32_ld_tnsr_b(phi, partial);

    wp[0] = g2; ap[2] = g2; plo[2] = g2; phi[2] = g2;
    float64 ws2 = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as2 = v_f32_ld_g(gen_addr(ap, activation_scales));
    float64 xlo2 = v_f32_ld_tnsr_b(plo, partial), xhi2 = v_f32_ld_tnsr_b(phi, partial);

    wp[0] = g3; ap[2] = g3; plo[2] = g3; phi[2] = g3;
    float64 ws3 = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as3 = v_f32_ld_g(gen_addr(ap, activation_scales));
    float64 xlo3 = v_f32_ld_tnsr_b(plo, partial), xhi3 = v_f32_ld_tnsr_b(phi, partial);

    float64 f0 = ws0 * as0, f1 = ws1 * as1, f2 = ws2 * as2, f3 = ws3 * as3;
    lo0 = v_f32_mac_b(xlo0, f0, lo0); hi0 = v_f32_mac_b(xhi0, f0, hi0);
    lo0 = v_f32_mac_b(xlo1, f1, lo0); hi0 = v_f32_mac_b(xhi1, f1, hi0);
    lo0 = v_f32_mac_b(xlo2, f2, lo0); hi0 = v_f32_mac_b(xhi2, f2, hi0);
    lo0 = v_f32_mac_b(xlo3, f3, lo0); hi0 = v_f32_mac_b(xhi3, f3, hi0);
  }
  for (; g < groups; ++g) {
    wp[0] = g; ap[2] = g; plo[2] = g; phi[2] = g;
    float64 f0 = v_f32_ld_g(gen_addr(wp, weight_scales)) * v_f32_ld_g(gen_addr(ap, activation_scales));
    lo0 = v_f32_mac_b(v_f32_ld_tnsr_b(plo, partial), f0, lo0);
    hi0 = v_f32_mac_b(v_f32_ld_tnsr_b(phi, partial), f0, hi0);
  }
  int5 p = {col, row, 0, 0, 0};
#if OUTPUT_BF16
  float128 pair = {lo0 + blo, hi0 + bhi};
  v_bf16_st_tnsr(p, output, convert_float128_to_bfloat128(pair, SW_LINEAR | SW_RHNE));
#else
  v_f32_st_tnsr(p, output, lo0); p[0] += 64; v_f32_st_tnsr(p, output, hi0);
#endif
}

static inline void reduce_2rows(
    tensor partial, tensor activation_scales, tensor weight_scales, tensor output,
    int col, int nb, int row, int groups, float64 blo, float64 bhi) {
  float64 lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
  int5 wp = {0, nb, 0, 0, 0};
  int5 ap0 = {0, row, 0, 0, 0}, ap1 = {0, row + 1, 0, 0, 0};
  int5 p0lo = {col, row, 0, 0, 0}, p0hi = {col + 64, row, 0, 0, 0};
  int5 p1lo = {col, row + 1, 0, 0, 0}, p1hi = {col + 64, row + 1, 0, 0, 0};
  int g2 = groups & ~1, g = 0;
  for (; g < g2; g += 2) {
    int g1 = g | 1;
    wp[0] = g; ap0[2] = g; ap1[2] = g;
    p0lo[2] = g; p0hi[2] = g; p1lo[2] = g; p1hi[2] = g;
    float64 ws_a = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as0_a = v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 as1_a = v_f32_ld_g(gen_addr(ap1, activation_scales));
    float64 x0lo_a = v_f32_ld_tnsr_b(p0lo, partial), x0hi_a = v_f32_ld_tnsr_b(p0hi, partial);
    float64 x1lo_a = v_f32_ld_tnsr_b(p1lo, partial), x1hi_a = v_f32_ld_tnsr_b(p1hi, partial);

    wp[0] = g1; ap0[2] = g1; ap1[2] = g1;
    p0lo[2] = g1; p0hi[2] = g1; p1lo[2] = g1; p1hi[2] = g1;
    float64 ws_b = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as0_b = v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 as1_b = v_f32_ld_g(gen_addr(ap1, activation_scales));
    float64 x0lo_b = v_f32_ld_tnsr_b(p0lo, partial), x0hi_b = v_f32_ld_tnsr_b(p0hi, partial);
    float64 x1lo_b = v_f32_ld_tnsr_b(p1lo, partial), x1hi_b = v_f32_ld_tnsr_b(p1hi, partial);

    float64 f0_a = ws_a * as0_a, f1_a = ws_a * as1_a;
    lo0 = v_f32_mac_b(x0lo_a, f0_a, lo0); hi0 = v_f32_mac_b(x0hi_a, f0_a, hi0);
    lo1 = v_f32_mac_b(x1lo_a, f1_a, lo1); hi1 = v_f32_mac_b(x1hi_a, f1_a, hi1);

    float64 f0_b = ws_b * as0_b, f1_b = ws_b * as1_b;
    lo0 = v_f32_mac_b(x0lo_b, f0_b, lo0); hi0 = v_f32_mac_b(x0hi_b, f0_b, hi0);
    lo1 = v_f32_mac_b(x1lo_b, f1_b, lo1); hi1 = v_f32_mac_b(x1hi_b, f1_b, hi1);
  }
  for (; g < groups; ++g) {
    wp[0] = g; ap0[2] = g; ap1[2] = g;
    p0lo[2] = g; p0hi[2] = g; p1lo[2] = g; p1hi[2] = g;
    float64 ws = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 f0 = ws * v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 f1 = ws * v_f32_ld_g(gen_addr(ap1, activation_scales));
    lo0 = v_f32_mac_b(v_f32_ld_tnsr_b(p0lo, partial), f0, lo0);
    hi0 = v_f32_mac_b(v_f32_ld_tnsr_b(p0hi, partial), f0, hi0);
    lo1 = v_f32_mac_b(v_f32_ld_tnsr_b(p1lo, partial), f1, lo1);
    hi1 = v_f32_mac_b(v_f32_ld_tnsr_b(p1hi, partial), f1, hi1);
  }
  int5 o0 = {col, row, 0, 0, 0}, o1 = {col, row + 1, 0, 0, 0};
#if OUTPUT_BF16
  float128 r0 = {lo0 + blo, hi0 + bhi}, r1 = {lo1 + blo, hi1 + bhi};
  v_bf16_st_tnsr(o0, output, convert_float128_to_bfloat128(r0, SW_LINEAR | SW_RHNE));
  v_bf16_st_tnsr(o1, output, convert_float128_to_bfloat128(r1, SW_LINEAR | SW_RHNE));
#else
  v_f32_st_tnsr(o0, output, lo0); o0[0] += 64; v_f32_st_tnsr(o0, output, hi0);
  v_f32_st_tnsr(o1, output, lo1); o1[0] += 64; v_f32_st_tnsr(o1, output, hi1);
#endif
}

static inline void reduce_4rows(
    tensor partial, tensor activation_scales, tensor weight_scales, tensor output,
    int col, int nb, int row, int groups, float64 blo, float64 bhi) {
  float64 lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0, lo2 = 0, hi2 = 0, lo3 = 0, hi3 = 0;
  int5 wp = {0, nb, 0, 0, 0};
  int5 ap0 = {0, row, 0, 0, 0}, ap1 = {0, row + 1, 0, 0, 0}, ap2 = {0, row + 2, 0, 0, 0}, ap3 = {0, row + 3, 0, 0, 0};
  int5 p0lo = {col, row, 0, 0, 0}, p0hi = {col + 64, row, 0, 0, 0};
  int5 p1lo = {col, row + 1, 0, 0, 0}, p1hi = {col + 64, row + 1, 0, 0, 0};
  int5 p2lo = {col, row + 2, 0, 0, 0}, p2hi = {col + 64, row + 2, 0, 0, 0};
  int5 p3lo = {col, row + 3, 0, 0, 0}, p3hi = {col + 64, row + 3, 0, 0, 0};
  int g2 = groups & ~1, g = 0;
  for (; g < g2; g += 2) {
    int g1 = g | 1;
    wp[0] = g; ap0[2] = g; ap1[2] = g; ap2[2] = g; ap3[2] = g;
    p0lo[2] = g; p0hi[2] = g; p1lo[2] = g; p1hi[2] = g;
    p2lo[2] = g; p2hi[2] = g; p3lo[2] = g; p3hi[2] = g;
    float64 ws_a = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as0_a = v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 as1_a = v_f32_ld_g(gen_addr(ap1, activation_scales));
    float64 as2_a = v_f32_ld_g(gen_addr(ap2, activation_scales));
    float64 as3_a = v_f32_ld_g(gen_addr(ap3, activation_scales));
    float64 x0lo_a = v_f32_ld_tnsr_b(p0lo, partial), x0hi_a = v_f32_ld_tnsr_b(p0hi, partial);
    float64 x1lo_a = v_f32_ld_tnsr_b(p1lo, partial), x1hi_a = v_f32_ld_tnsr_b(p1hi, partial);
    float64 x2lo_a = v_f32_ld_tnsr_b(p2lo, partial), x2hi_a = v_f32_ld_tnsr_b(p2hi, partial);
    float64 x3lo_a = v_f32_ld_tnsr_b(p3lo, partial), x3hi_a = v_f32_ld_tnsr_b(p3hi, partial);

    wp[0] = g1; ap0[2] = g1; ap1[2] = g1; ap2[2] = g1; ap3[2] = g1;
    p0lo[2] = g1; p0hi[2] = g1; p1lo[2] = g1; p1hi[2] = g1;
    p2lo[2] = g1; p2hi[2] = g1; p3lo[2] = g1; p3hi[2] = g1;
    float64 ws_b = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 as0_b = v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 as1_b = v_f32_ld_g(gen_addr(ap1, activation_scales));
    float64 as2_b = v_f32_ld_g(gen_addr(ap2, activation_scales));
    float64 as3_b = v_f32_ld_g(gen_addr(ap3, activation_scales));
    float64 x0lo_b = v_f32_ld_tnsr_b(p0lo, partial), x0hi_b = v_f32_ld_tnsr_b(p0hi, partial);
    float64 x1lo_b = v_f32_ld_tnsr_b(p1lo, partial), x1hi_b = v_f32_ld_tnsr_b(p1hi, partial);
    float64 x2lo_b = v_f32_ld_tnsr_b(p2lo, partial), x2hi_b = v_f32_ld_tnsr_b(p2hi, partial);
    float64 x3lo_b = v_f32_ld_tnsr_b(p3lo, partial), x3hi_b = v_f32_ld_tnsr_b(p3hi, partial);

    float64 f0_a = ws_a * as0_a, f1_a = ws_a * as1_a, f2_a = ws_a * as2_a, f3_a = ws_a * as3_a;
    lo0 = v_f32_mac_b(x0lo_a, f0_a, lo0); hi0 = v_f32_mac_b(x0hi_a, f0_a, hi0);
    lo1 = v_f32_mac_b(x1lo_a, f1_a, lo1); hi1 = v_f32_mac_b(x1hi_a, f1_a, hi1);
    lo2 = v_f32_mac_b(x2lo_a, f2_a, lo2); hi2 = v_f32_mac_b(x2hi_a, f2_a, hi2);
    lo3 = v_f32_mac_b(x3lo_a, f3_a, lo3); hi3 = v_f32_mac_b(x3hi_a, f3_a, hi3);

    float64 f0_b = ws_b * as0_b, f1_b = ws_b * as1_b, f2_b = ws_b * as2_b, f3_b = ws_b * as3_b;
    lo0 = v_f32_mac_b(x0lo_b, f0_b, lo0); hi0 = v_f32_mac_b(x0hi_b, f0_b, hi0);
    lo1 = v_f32_mac_b(x1lo_b, f1_b, lo1); hi1 = v_f32_mac_b(x1hi_b, f1_b, hi1);
    lo2 = v_f32_mac_b(x2lo_b, f2_b, lo2); hi2 = v_f32_mac_b(x2hi_b, f2_b, hi2);
    lo3 = v_f32_mac_b(x3lo_b, f3_b, lo3); hi3 = v_f32_mac_b(x3hi_b, f3_b, hi3);
  }
  for (; g < groups; ++g) {
    wp[0] = g; ap0[2] = g; ap1[2] = g; ap2[2] = g; ap3[2] = g;
    p0lo[2] = g; p0hi[2] = g; p1lo[2] = g; p1hi[2] = g;
    p2lo[2] = g; p2hi[2] = g; p3lo[2] = g; p3hi[2] = g;
    float64 ws = v_f32_ld_g(gen_addr(wp, weight_scales));
    float64 f0 = ws * v_f32_ld_g(gen_addr(ap0, activation_scales));
    float64 f1 = ws * v_f32_ld_g(gen_addr(ap1, activation_scales));
    float64 f2 = ws * v_f32_ld_g(gen_addr(ap2, activation_scales));
    float64 f3 = ws * v_f32_ld_g(gen_addr(ap3, activation_scales));
    lo0 = v_f32_mac_b(v_f32_ld_tnsr_b(p0lo, partial), f0, lo0);
    hi0 = v_f32_mac_b(v_f32_ld_tnsr_b(p0hi, partial), f0, hi0);
    lo1 = v_f32_mac_b(v_f32_ld_tnsr_b(p1lo, partial), f1, lo1);
    hi1 = v_f32_mac_b(v_f32_ld_tnsr_b(p1hi, partial), f1, hi1);
    lo2 = v_f32_mac_b(v_f32_ld_tnsr_b(p2lo, partial), f2, lo2);
    hi2 = v_f32_mac_b(v_f32_ld_tnsr_b(p2hi, partial), f2, hi2);
    lo3 = v_f32_mac_b(v_f32_ld_tnsr_b(p3lo, partial), f3, lo3);
    hi3 = v_f32_mac_b(v_f32_ld_tnsr_b(p3hi, partial), f3, hi3);
  }
  int5 o0 = {col, row, 0, 0, 0}, o1 = {col, row + 1, 0, 0, 0}, o2 = {col, row + 2, 0, 0, 0}, o3 = {col, row + 3, 0, 0, 0};
#if OUTPUT_BF16
  float128 r0 = {lo0 + blo, hi0 + bhi}, r1 = {lo1 + blo, hi1 + bhi}, r2 = {lo2 + blo, hi2 + bhi}, r3 = {lo3 + blo, hi3 + bhi};
  v_bf16_st_tnsr(o0, output, convert_float128_to_bfloat128(r0, SW_LINEAR | SW_RHNE));
  v_bf16_st_tnsr(o1, output, convert_float128_to_bfloat128(r1, SW_LINEAR | SW_RHNE));
  v_bf16_st_tnsr(o2, output, convert_float128_to_bfloat128(r2, SW_LINEAR | SW_RHNE));
  v_bf16_st_tnsr(o3, output, convert_float128_to_bfloat128(r3, SW_LINEAR | SW_RHNE));
#else
  v_f32_st_tnsr(o0, output, lo0); o0[0] += 64; v_f32_st_tnsr(o0, output, hi0);
  v_f32_st_tnsr(o1, output, lo1); o1[0] += 64; v_f32_st_tnsr(o1, output, hi1);
  v_f32_st_tnsr(o2, output, lo2); o2[0] += 64; v_f32_st_tnsr(o2, output, hi2);
  v_f32_st_tnsr(o3, output, lo3); o3[0] += 64; v_f32_st_tnsr(o3, output, hi3);
#endif
}

void main(tensor partial, tensor activation_scales, tensor weight_scales, tensor bias, tensor output) {
  int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
  int groups = get_dim_size(partial, 2), rows = get_dim_size(output, 1);
#if ROWS == 4
  if (rows <= MAX_1ROW) {
    for (int nb = begin[0]; nb < end[0]; ++nb) {
      int col = nb * 128;
      int5 b0 = {col, 0, 0, 0, 0}, b1 = {col + 64, 0, 0, 0, 0};
      float64 blo = v_f32_ld_tnsr_b(b0, bias), bhi = v_f32_ld_tnsr_b(b1, bias);
      for (int rb = begin[1]; rb < end[1]; ++rb) {
        reduce_1row(partial, activation_scales, weight_scales, output, col, nb, rb, groups, blo, bhi);
      }
    }
  } else if (rows <= MAX_2ROW) {
    for (int rb = begin[1]; rb < end[1]; ++rb) {
      int row = rb * 2;
      for (int nb = begin[0]; nb < end[0]; ++nb) {
        int col = nb * 128;
        int5 b0 = {col, 0, 0, 0, 0}, b1 = {col + 64, 0, 0, 0, 0};
        float64 blo = v_f32_ld_tnsr_b(b0, bias), bhi = v_f32_ld_tnsr_b(b1, bias);
        if (row + 1 < rows) {
          reduce_2rows(partial, activation_scales, weight_scales, output, col, nb, row, groups, blo, bhi);
        } else {
          reduce_1row(partial, activation_scales, weight_scales, output, col, nb, row, groups, blo, bhi);
        }
      }
    }
  } else {
    for (int rb = begin[1]; rb < end[1]; ++rb) {
      int row = rb * 4;
      for (int nb = begin[0]; nb < end[0]; ++nb) {
        int col = nb * 128;
        int5 b0 = {col, 0, 0, 0, 0}, b1 = {col + 64, 0, 0, 0, 0};
        float64 blo = v_f32_ld_tnsr_b(b0, bias), bhi = v_f32_ld_tnsr_b(b1, bias);
        if (row + 3 < rows) {
          reduce_4rows(partial, activation_scales, weight_scales, output, col, nb, row, groups, blo, bhi);
        } else {
          for (int r = row; r < rows; ++r) {
            reduce_1row(partial, activation_scales, weight_scales, output, col, nb, r, groups, blo, bhi);
          }
        }
      }
    }
  }
#else
  for (int rb = begin[1]; rb < end[1]; ++rb) {
    for (int nb = begin[0]; nb < end[0]; ++nb) {
      int col = nb * 128;
      int5 b0 = {col, 0, 0, 0, 0}, b1 = {col + 64, 0, 0, 0, 0};
      float64 blo = v_f32_ld_tnsr_b(b0, bias), bhi = v_f32_ld_tnsr_b(b1, bias);
      reduce_1row(partial, activation_scales, weight_scales, output, col, nb, rb, groups, blo, bhi);
    }
  }
#endif
}

