// OCP E4M3FN activation quantization followed by Gaudi2 half-range encoding.
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
