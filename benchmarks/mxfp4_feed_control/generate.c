// Synthetic endpoint control: seed -> SRAM. No MXFP4 load or decode.
// Alternating signs prevent a constant K tensor; experts have separate seed regions.
#ifdef FEED_FP8
#define LANES 256
#define VECTOR uchar256
#define LOAD v_u8_ld_tnsr_b
#define STORE v_u8_st_tnsr
#define SIGN 128
#else
#define LANES 128
#define VECTOR ushort128
#define LOAD v_u16_ld_tnsr_b
#define STORE v_u16_st_tnsr
#define SIGN 32768
#endif
#ifdef FEED_FENCE
void main(tensor seed, tensor fence, tensor weight) {
    int5 fp = {0,0,0,0,0};
    float signal = s_f32_ld_g(gen_addr(fp,fence));
    bool ready = signal == signal; // Real data dependence; reject a NaN predecessor.
#else
void main(tensor seed, tensor weight) {
    bool ready = 1;
#endif
    int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
    for (int e = begin[2]; e < end[2]; ++e)
        for (int n = begin[0]; n < end[0]; ++n) {
            int5 s = {n * LANES, e, 0, 0, 0};
            VECTOR positive = LOAD(s, seed);
            VECTOR negative = positive ^ SIGN;
            for (int group = begin[1]; group < end[1]; ++group) {
                int5 p = {n * LANES, group * 32, e, 0, 0};
                for (int k = 0; k < 32; k += 4) {
                    STORE(p, weight, positive, 0, ready); p[1] += 1;
                    STORE(p, weight, negative, 0, ready); p[1] += 1;
                    STORE(p, weight, positive, 0, ready); p[1] += 1;
                    STORE(p, weight, negative, 0, ready); p[1] += 1;
                }
            }
        }
}
