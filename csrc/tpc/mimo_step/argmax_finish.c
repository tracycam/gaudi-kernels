// SPDX-License-Identifier: Apache-2.0
// -1 reports either a nonfinite logit or an invalid embedding token.
void main(tensor partial_max,tensor partial_id,tensor partial_bad,tensor token_status,tensor next_token) {
 int5 p={0,0,0,0,0};
 int invalid=s_i32_ld_g(gen_addr(p,token_status));
 float best=-1.0f/0.0f;int chosen=2147483647;
 for(int part=0;part<24;++part){
  p[0]=part;
  float value=s_f32_ld_g(gen_addr(p,partial_max));
  int id=s_i32_ld_g(gen_addr(p,partial_id));
  invalid|=s_i32_ld_g(gen_addr(p,partial_bad));
  if(value>best || (value==best && id<chosen)){best=value;chosen=id;}
 }
 p[0]=0;
 s_i32_st_g(gen_addr(p,next_token),invalid ? -1 : chosen);
}
