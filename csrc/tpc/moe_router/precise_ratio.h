// Normalize both mantissas before residual correction. Correcting an unscaled
// tiny quotient can lose the residual to FTZ (retained random-11 counterexample).
// This uses FP32 throughout and explicitly retains native FTZ/special classes.
static inline float64 router_precise_ratio(float64 numerator,float64 denominator){
 uint64 nb=(uint64)numerator,db=(uint64)denominator;
 uint64 n=nb&0x7fffffff,d=db&0x7fffffff,sign=(nb^db)&0x80000000u;
 float64 nm=(float64)((n&0x7fffff)|0x3f800000u);
 float64 dm=(float64)((d&0x7fffff)|0x3f800000u);
 float64 reciprocal=v_reciprocal_f32(dm);
 float64 quotient=nm*reciprocal;
 float64 residual=v_f32_mac_b(-quotient,dm,nm);
 quotient=v_f32_mac_b(residual,reciprocal,quotient);
 uint64 qb=(uint64)quotient;
 int64 exponent=(int64)(qb>>23)+(int64)(n>>23)-(int64)(d>>23);
 uint64 bits=(qb&0x7fffff)|((uint64)exponent<<23)|sign;
 bits=v_u32_sel_less_i32_b(exponent,1,sign,bits);
 bits=v_u32_sel_geq_i32_b(exponent,255,sign|0x7f800000u,bits);
 bits=v_u32_sel_less_u32_b(n,0x00800000u,sign,bits);
 bits=v_u32_sel_eq_u32_b(d,0x7f800000u,sign,bits);
 bits=v_u32_sel_eq_u32_b(n,0x7f800000u,sign|0x7f800000u,bits);
 uint64 when_zero=v_u32_sel_less_u32_b(n,0x00800000u,0x7fffffffu,sign|0x7f800000u);
 bits=v_u32_sel_less_u32_b(d,0x00800000u,when_zero,bits);
 uint64 when_inf=v_u32_sel_eq_u32_b(n,0x7f800000u,0x7fffffffu,bits);
 bits=v_u32_sel_eq_u32_b(d,0x7f800000u,when_inf,bits);
 bits=v_u32_sel_grt_u32_b(n,0x7f800000u,0x7fffffffu,bits);
 bits=v_u32_sel_grt_u32_b(d,0x7f800000u,0x7fffffffu,bits);
 return (float64)bits;
}
