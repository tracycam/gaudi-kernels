// Bounded dependency chain reused from reduction-floor-20260924/kernel.c.
// Timing begins AFTER this kernel; synEventQuery must prove work was prequeued.
void main(tensor input, tensor output, int loops) {
 int5 p={0,0,0,0,0};
 float64 x=v_f32_ld_tnsr_b(p,input);
 for(int j=0;j<loops;++j){
  x=v_f32_mul_b(x,0.99999f);
  x=v_f32_add_b(x,0.00001f);
 }
 v_f32_st_tnsr(p,output,x);
}
