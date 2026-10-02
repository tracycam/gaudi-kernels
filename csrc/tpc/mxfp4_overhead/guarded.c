// N512 byte-pair layout. Four independent BF16 MAC -> FP32 chains.
// Compile EXACT=1 to force the original scaled-weight arithmetic comparator.
#ifndef EXACT
#define EXACT 0
#endif
static inline bfloat128 scale_exact(bfloat128 q,ushort128 exponent){
 // The admitted scale domain makes q*scale exactly representable normal BF16.
 // FMA with negative zero preserves signed zero (same mechanism as 1cat's
 // prepared decoder) without the per-weight integer exponent fixup chain.
 bfloat128 scale=(bfloat128)((exponent&255)<<7);
 bfloat128 negative_zero=(bfloat128)((ushort128)32768);
 return v_bf16_madd_b(q,scale,negative_zero);
}
static inline bfloat128 scale_subnormal(bfloat128 q,ushort128 exponent){
 // x was multiplied by 256 exactly. Subtract 8 from its weight exponent.
 // Codes <10 produce terms <2^-240 even before compensation; even int-sized
 // K cannot make their sum representable in FP32, so zero those lanes.
 q=(bfloat128)v_u16_sel_less_u16_b(exponent,10,0,(ushort128)q);
 exponent=v_u16_sel_less_u16_b(exponent,10,10,exponent)-8;
 return scale_exact(q,exponent);
}
void main(tensor packed,tensor scales,tensor activation,tensor table,tensor mapping,tensor unsafe,tensor output){
 set_lut_256(table);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(activation,0);
 for(int task=begin[0];task<end[0];++task){
  int5 mp={task,0,0,0,0},fp={0,task,0,0,0};
  int mapped=s_i32_ld_g(gen_addr(mp,mapping));
  bool exact=EXACT||s_u16_ld_g(gen_addr(fp,unsafe));
  float128 t0={0},t1={0},t2={0},t3={0};
  for(int g=0;g<K/32;++g){
   int5 sc={0,mapped*(K/32)+g,0,0,0};
   uchar256 e0=v_u8_ld_tnsr_b(sc,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sc[0]=256;uchar256 e1=v_u8_ld_tnsr_b(sc,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
   ushort256 s0=convert_uchar256_to_ushort256(e0,0),s1=convert_uchar256_to_ushort256(e1,0);
   float128 p0={0},p1={0},p2={0},p3={0};
   for(int j=0;j<32;++j){
    int5 w={0,mapped*K+g*32+j,0,0,0},xp={g*32+j,task,0,0,0};
    uchar256 bytes=v_u8_ld_tnsr_b(w,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
    ushort256 indices=convert_uchar256_to_ushort256(bytes,0);
    bfloat256 a=v_bf16_lookup_2c(indices.v1,0,SW_LUT_PTR,(bfloat256){0});
    bfloat256 b=v_bf16_lookup_2c(indices.v2,0,SW_LUT_PTR,(bfloat256){0});
    bf16 x=s_bf16_ld_g(gen_addr(xp,activation));
    if(exact){
     unsigned short bits=as_ushort(x),mantissa=bits&32767;
     if(mantissa>0 && mantissa<128){
      // BF16 MAC flushes subnormal operands. Bit-normalize x*256 exactly;
      // this changes neither activation precision nor the represented product.
      unsigned short exponent=9;
      while(mantissa<128){mantissa<<=1;--exponent;}
      x=as_bf16((unsigned short)((bits&32768)|(exponent<<7)|(mantissa&127)));
      t0=v_bf16_mac_acc32_b(scale_subnormal(a.v1,s0.v1),x,t0);
      t1=v_bf16_mac_acc32_b(scale_subnormal(a.v2,s0.v2),x,t1);
      t2=v_bf16_mac_acc32_b(scale_subnormal(b.v1,s1.v1),x,t2);
      t3=v_bf16_mac_acc32_b(scale_subnormal(b.v2,s1.v2),x,t3);
     }else{
     t0=v_bf16_mac_acc32_b(scale_exact(a.v1,s0.v1),x,t0);
     t1=v_bf16_mac_acc32_b(scale_exact(a.v2,s0.v2),x,t1);
     t2=v_bf16_mac_acc32_b(scale_exact(b.v1,s1.v1),x,t2);
     t3=v_bf16_mac_acc32_b(scale_exact(b.v2,s1.v2),x,t3);
     }
    }else{
     p0=v_bf16_mac_acc32_b(a.v1,x,p0);p1=v_bf16_mac_acc32_b(a.v2,x,p1);
     p2=v_bf16_mac_acc32_b(b.v1,x,p2);p3=v_bf16_mac_acc32_b(b.v2,x,p3);
    }
   }
   if(!exact){
    float128 f0=v_convert_bf16_to_f32_all_b((bfloat128)(s0.v1<<7));
    float128 f1=v_convert_bf16_to_f32_all_b((bfloat128)(s0.v2<<7));
    float128 f2=v_convert_bf16_to_f32_all_b((bfloat128)(s1.v1<<7));
    float128 f3=v_convert_bf16_to_f32_all_b((bfloat128)(s1.v2<<7));
    t0.v1=v_f32_mac_b(p0.v1,f0.v1,t0.v1);t0.v2=v_f32_mac_b(p0.v2,f0.v2,t0.v2);
    t1.v1=v_f32_mac_b(p1.v1,f1.v1,t1.v1);t1.v2=v_f32_mac_b(p1.v2,f1.v2,t1.v2);
    t2.v1=v_f32_mac_b(p2.v1,f2.v1,t2.v1);t2.v2=v_f32_mac_b(p2.v2,f2.v2,t2.v2);
    t3.v1=v_f32_mac_b(p3.v1,f3.v1,t3.v1);t3.v2=v_f32_mac_b(p3.v2,f3.v2,t3.v2);
   }
  }
  int5 o={task*512,0,0,0,0};
  v_f32_st_tnsr(o,output,t0.v1);o[0]+=64;v_f32_st_tnsr(o,output,t0.v2);o[0]+=64;
  v_f32_st_tnsr(o,output,t1.v1);o[0]+=64;v_f32_st_tnsr(o,output,t1.v2);o[0]+=64;
  v_f32_st_tnsr(o,output,t2.v1);o[0]+=64;v_f32_st_tnsr(o,output,t2.v2);o[0]+=64;
  v_f32_st_tnsr(o,output,t3.v1);o[0]+=64;v_f32_st_tnsr(o,output,t3.v2);
 }
}
