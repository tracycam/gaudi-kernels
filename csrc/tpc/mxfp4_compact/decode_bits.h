// Integer BF16 bit manipulation only. All E8M0 codes MUST be in [2,252].
// General finite MXFP4 scales [0,254] must use the separate exact reader.
#ifndef GK_MXFP4_COMPACT_DECODE_BITS_H
#define GK_MXFP4_COMPACT_DECODE_BITS_H
static inline bfloat128 compact_scaled(bfloat128 q, ushort128 exponent) {
    ushort128 bits=(ushort128)q, mag=bits&32767;
#if defined(GK_ROUTE_DECODE_SCALE_BITS) && GK_ROUTE_DECODE_SCALE_BITS
    // Nonzero E2M1 BF16 exponents are 126..129. Legal scale 2..252
    // produces exponents 1..254, so integer addition cannot cross the sign.
    // Select the original bits for either zero; do not erase negative zero.
    ushort128 adjusted=bits+(((exponent&255)-127)<<7);
    return (bfloat128)v_u16_sel_eq_u16_b(mag,0,bits,adjusted);
#else
    ushort128 adjusted=(mag+(((exponent&255)-127)<<7))&32767;
    adjusted=v_u16_sel_eq_u16_b(mag,0,0,adjusted);
    return (bfloat128)(adjusted|(bits&32768));
#endif
}
// Native-v2 keeps even/odd physical lanes interleaved for its BF16 MAC.
// Widen in hardware even/odd order, then narrow in linear order to restore N.
static inline bfloat128 compact_native_to_logical(bfloat128 q) {
    uint128 expanded=v_convert_u16_to_u32_all_b((ushort128)q);
    return (bfloat128)convert_uint128_to_ushort128(expanded,SW_LINEAR);
}
// First 16 low/high nibbles become first 32 logical K values. No FP conversion.
static inline bfloat128 compact_interleave_k(bfloat128 low,bfloat128 high) {
    uint128 l=convert_ushort128_to_uint128((ushort128)low,SW_LINEAR);
    uint128 h=convert_ushort128_to_uint128((ushort128)high,SW_LINEAR);
    uint128 pair={l.v1,h.v1};
    return (bfloat128)v_convert_u32_to_u16_all_b(pair);
}
#endif
