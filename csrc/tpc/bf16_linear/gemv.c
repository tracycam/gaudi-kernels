// Offline transposed BF16 weights [N,K], read once for up to two token rows.
// Exact BF16->FP32 promotion is register-local; FP32 FMA and split-K reduction.
#ifndef ROWS
#define ROWS 1
#endif
void main(tensor activation,tensor weight,tensor partial) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(activation,0);
 for(int g=begin[1];g<end[1];++g)for(int nb=begin[0];nb<end[0];++nb){
  #if NATIVE_MAC
  float128 acc0={0,0};
#else
  float64 lo0=0,hi0=0;
#endif
#if ROWS == 2
  #if NATIVE_MAC
  float128 acc1={0,0};
#else
  float64 lo1=0,hi1=0;
#endif
#endif
  int5 wp={nb*128,g*256,0,0,0},xp={g*256,0,0,0,0};
  int count=K-g*256<256?K-g*256:256;
  #pragma loop_unroll(4)
  for(int z=0;z<count;++z){
   bfloat128 packed=v_bf16_ld_tnsr_b(wp,weight);
   #if NATIVE_MAC
   xp[1]=0;bf16 a0=s_bf16_ld_g(gen_addr(xp,activation));acc0=v_bf16_mac_acc32_b(packed,a0,acc0);
#if ROWS == 2
   xp[1]=1;bf16 a1=s_bf16_ld_g(gen_addr(xp,activation));acc1=v_bf16_mac_acc32_b(packed,a1,acc1);
#endif
#else
   float128 w=convert_bfloat128_to_float128(packed,SW_LINEAR);
   xp[1]=0;float a0=(float)s_bf16_ld_g(gen_addr(xp,activation));
   lo0=v_f32_mac_b(w.v1,a0,lo0);hi0=v_f32_mac_b(w.v2,a0,hi0);
#if ROWS == 2
   xp[1]=1;float a1=(float)s_bf16_ld_g(gen_addr(xp,activation));
   lo1=v_f32_mac_b(w.v1,a1,lo1);hi1=v_f32_mac_b(w.v2,a1,hi1);
#endif
   #endif
   wp[1]++;xp[0]++;
  }
#if NATIVE_MAC
#define lo0 acc0.v1
#define hi0 acc0.v2
#define lo1 acc1.v1
#define hi1 acc1.v2
#endif
#define STORE(R) {int5 p={nb*128,R,g,0,0};v_f32_st_tnsr(p,partial,lo##R);p[0]+=64;v_f32_st_tnsr(p,partial,hi##R);}
  STORE(0)
#if ROWS == 2
  STORE(1)
#endif
#undef STORE
 }
}
