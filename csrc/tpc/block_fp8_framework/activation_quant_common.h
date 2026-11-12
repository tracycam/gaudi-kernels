// OCP E4M3FN activation quantization helpers (half-range encoding for 1-byte MME
// and in-register OCP E4M3FN quantization for SRAM-bundled 1K FP16 MME).
static inline float64 fp8_activation_scale(float64 maximum) {
 float64 scale=maximum*(1.0f/448.0f);
 float64 residual=v_f32_mac_b(-scale,448.0f,maximum);
 return v_f32_mac_b(residual,(1.0f/448.0f),scale);
}

static inline float64 fp8_quant_half(float64 x, float64 scale, float64 reciprocal) {
 float64 q=x*reciprocal;
 float64 residual=v_f32_mac_b(-q,scale,x);
 float64 z=v_f32_mac_b(residual,reciprocal,q);
 float64 magnitude=(float64)((uint64)z & 0x7fffffff);
 magnitude=v_f32_min_b(magnitude,448.0f);
 uint64 bits=(uint64)magnitude;
 uint64 rounded=(bits+0x7ffff+((bits>>20)&1)) & 0xfff00000;
 float64 normal=(float64)rounded;
 float64 subnormal=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(magnitude*512.0f,SW_RHNE))*(1.0f/512.0f);
 float64 value=v_f32_sel_less_f32_b(magnitude,0.015625f,subnormal,normal);
 return (float64)((uint64)(value*0.5f) | ((uint64)x & 0x80000000));
}

static inline minifloat256 fp8_pack_half(float128 x, float64 scale, float64 reciprocal) {
 float128 q; q.v1=fp8_quant_half(x.v1,scale,reciprocal); q.v2=fp8_quant_half(x.v2,scale,reciprocal);
 bfloat128 bf=convert_float128_to_bfloat128(q,SW_LINEAR|SW_RHNE);
 minifloat256 sparse=v_convert_bf16_to_f8_b(bf,0,SW_RHNE|SW_CLIP_FP,(minifloat256)0);
 minifloat256 packed=v_f8_pack_b(sparse,SW_GROUP_0|SW_STRIDE_2,(minifloat256)0);
 packed=v_f8_pack_b(sparse,SW_GROUP_1|SW_STRIDE_2,packed);
 return v_f8_mov_dual_group_pack_b(packed,SW_PACK21,(minifloat256)0);
}

// Positive FP32 reciprocal with conditional subnormal/near-subnormal prescaling
// (< 1e-30f) and two FMA Newton-Raphson steps, avoiding both subnormal reciprocal
// underflow on tiny positive scales and overflow on large scales.
static inline float64 fp8_exact_positive_rcp(float64 s) {
 float64 up=v_f32_sel_less_f32_b(s,1.0e-30f,16777216.0f,1.0f);
 float64 s_s=s*up;
 float64 r=v_reciprocal_fast_f32(s_s);
 r=v_f32_mac_b(v_f32_mac_b(-s_s,r,1.0f),r,r);
 r=v_f32_mac_b(v_f32_mac_b(-s_s,r,1.0f),r,r);
 return r*up;
}

// Exact OCP E4M3FN round-to-nearest-even code value in [-448, 448] in native
// BF16 even/odd lane order (preserving full 2^-9 subnormal grid without FTZ).
static inline float64 fp8_ocp_quant_val_vec(float64 x, float64 scale, float64 reciprocal) {
 float64 q=x*reciprocal;
 float64 residual=v_f32_mac_b(-q,scale,x);
 float64 z=v_f32_mac_b(residual,reciprocal,q);
 float64 magnitude=(float64)((uint64)z & 0x7fffffffu);
 magnitude=v_f32_min_b(magnitude,448.0f);
 uint64 bits=(uint64)magnitude;
 uint64 rounded=(bits+0x7ffffu+((bits>>20)&1u)) & 0xfff00000u;
 float64 normal=(float64)rounded;
 float64 subnormal=v_f32_sub_b(v_f32_add_b(magnitude,16384.0f),16384.0f);
 float64 value=v_f32_sel_less_f32_b(magnitude,0.015625f,subnormal,normal);
 return (float64)((uint64)value | ((uint64)x & 0x80000000u));
}

// Round positive target >= 0 up to an exact power of 2 in [2^-120, 2^+120]
// (returning 1.0f when raw_amax == 0.0f) so 1.0f / pow2_s is bit-exact.
static inline float64 fp8_pow2_ceil_scale(float64 target, float64 raw_amax) {
 float64 clamped=v_f32_min_b(v_f32_max_b(target,7.52316384526264e-37f),1.3292279957849159e36f);
 uint64 bits=((uint64)clamped+0x007fffffu) & 0x7f800000u;
 return v_f32_sel_eq_f32_b(raw_amax,0.0f,1.0f,(float64)bits);
}

// Exact IEEE754 reciprocal of a finite positive power-of-2 float in [2^-126, 2^+126].
static inline float64 fp8_pow2_exact_rcp(float64 pow2_s) {
 return (float64)(0x7f000000u-(uint64)pow2_s);
}
