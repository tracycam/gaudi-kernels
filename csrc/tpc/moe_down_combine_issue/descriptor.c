// Metadata scaffold only. Scheduled .text replaces this executable body.
// No private VLM staging; LUT setup is the same table[3] contract as production.
void main(tensor packed,tensor scales,tensor activation,tensor lut,tensor ids,
          tensor routing,tensor directions,tensor output) {
 set_lut_256(lut);int5 p={0,0,0,0,0};
 uchar256 raw=v_u8_ld_tnsr_b(p,packed);
 ushort256 indices=convert_uchar256_to_ushort256(raw,SW_LINEAR);
 bfloat256 decoded=v_bf16_lookup_2c(indices.v1,0,SW_LUT_PTR,(bfloat256){0});
 float128 lookup_used=v_convert_bf16_to_f32_all_b(decoded.v1);
 float64 v=lookup_used.v1+v_f32_ld_tnsr_b(p,packed)+v_f32_ld_tnsr_b(p,scales)+v_f32_ld_tnsr_b(p,activation);
 v+=v_f32_ld_tnsr_b(p,directions)+(float)s_i32_ld_g(gen_addr(p,ids))+s_f32_ld_g(gen_addr(p,routing));
 v_f32_st_tnsr(p,output,v);
}
