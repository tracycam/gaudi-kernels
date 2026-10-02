"""Adapt count-specialized N256 bodies to original HistoricalN512 owners."""
import importlib.util
from pathlib import Path


def wide_one(down, n, window):
    s='int5 rm={0,expert,0,0,0};int route=s_i32_ld_g(gen_addr(rm,row_map));int token=route'+(';'if down else'>>3;')
    s+=''.join(f'float128 total{j}={{0}};'for j in range(4))
    s+='for(int g=split*chunk;g<(split+1)*chunk&&g<groups;++g){'
    s+=''.join(f'float128 part{j}={{0}};'for j in range(4))
    s+='int5 xp={g*32,token,0,0,0};uchar256 xv=(uchar256)v_bf16_ld_tnsr_partial_b(xp,activation,31,0);xv=v_u8_mov_dual_group_all_b(xv,-1,0,0,0,0,MkWrA(3,3,3,3),0);'
    s+=f'\n#pragma loop_unroll(1)\nfor(int base=0;base<32;base+={window})'+'{'
    for z in range(window):
        s+='{'+f'int5 p={{0,(source*{n//512}+pair)*K+g*32+base+{z},0,0,0}};'
        s+='uchar256 raw=v_u8_ld_tnsr_b(p,packed);ushort256 ind=convert_uchar256_to_ushort256(raw,SW_LINEAR);'
        s+='bfloat256 q0=v_bf16_lookup_2c(ind.v1,0,SW_LUT_PTR,(bfloat256){0}),q1=v_bf16_lookup_2c(ind.v2,0,SW_LUT_PTR,(bfloat256){0});'
        s+=f'uchar256 mask=(uchar256)v_u32_mov_b(0x81808180u+(base+{z})*0x02020202u);bfloat128 x=(bfloat128)v_u8_shuffle_b(xv,mask,0,0);'
        s+=''.join(f'part{j}=v_bf16_mac_acc32_b(q{j//2}.v{j%2+1},x,part{j});'for j in range(4))+'}'
    s+='}'
    for j in range(4):
        s+='{'+f'int5 sp={{{j*128},(source*{n//512}+pair)*(K/32)+g,0,0,0}};ushort128 exp=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);'
        s+='float128 f=v_convert_bf16_to_f32_all_b((bfloat128)((exp&255)<<7));'
        s+=f'total{j}.v1=v_f32_mac_b(part{j}.v1,f.v1,total{j}.v1);total{j}.v2=v_f32_mac_b(part{j}.v2,f.v2,total{j}.v2);'+'}'
    s+='}'
    for j in range(4):
        s+='{'+f'float128 y=linear_acc(total{j});int5 o={{pair*512+{j*128},split,route,0,0}};v_f32_st_tnsr(o,output,y.v1);o[0]+=64;v_f32_st_tnsr(o,output,y.v2);'+'}'
    return s


def generate(root: Path, cap: int, down: bool, dense=False, stripe=False, wide_m1=False, wide_window=4):
    assert cap in (4,8)
    assert not wide_m1 or dense
    spec=importlib.util.spec_from_file_location('n256_body',root/'benchmarks/mxfp4_n128/generate.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    k,n,splits=(256,6144,1)if down else(6144,512,3)
    source=f'''#define main unused_legacy_main
#include "{root/'csrc/tpc/mxfp4_linear/gemv.c'}"
#undef main
void main(tensor packed,tensor scales,tensor table,tensor activation,tensor ids,tensor counts,tensor row_map,tensor inverse,tensor output){{
 set_lut_256(table);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 const int K={k},groups={k//32},chunk={k//32//splits};
 {f'volatile float64 totals[{4*cap}];' if cap==8 else ''}
 for(int split=begin[2];split<end[2];++split)
 for(int expert=begin[1];expert<end[1];++expert){{
  int5 ep={{expert,0,0,0,0}};
  int source=s_i32_ld_g(gen_addr(ep,ids)),count=s_i32_ld_g(gen_addr(ep,counts));
  if(source<0||source>=384||count<0||count>{cap})count=0;
  for(int nb=begin[0];nb<end[0];++nb){{
'''
    if dense:
        own='(expert%tokens)*8+expert/tokens' if stripe else 'expert'
        source+=f'''if(count==0){{
 int tokens=get_dim_size(counts,0)/8,own={own};int5 ip={{own,0,0,0,0}};
 if(s_i32_ld_g(gen_addr(ip,inverse))<0){{
  float64 nan=(float64)(uint64)0x7fc00000;int5 o={{nb*256,split,own,0,0}};
  for(int j=0;j<4;++j){{v_f32_st_tnsr(o,output,nan);o[0]+=64;}}
 }}
 continue;
}}
'''
    if wide_m1:
        source=source.replace('for(int nb=begin[0];nb<end[0];++nb){','for(int pair=begin[0];pair<end[0];++pair){int nb=pair*2;')
        source=source.replace('j<4;++j','j<8;++j')
        source+='if(count==1){'+wide_one(down,n,wide_window)+'}else{for(int nsub=0;nsub<2;++nsub){nb=pair*2+nsub;'
    first=2 if wide_m1 else 0
    for m in range(first,cap+1):
        source+=('if'if m==first else'else if')+f'(count=={m}){{\n'
        # Eight-row N256 bodies exceed the compiler's 16 KiB spill allocation
        # with a K8 window. Reduce only high-M live ranges; M1..4 stay K8.
        body=mod.body(m,m if dense else cap,4 if m>=5 else 8,tile=256)
        if m>=5:
            # The total FP32 sum is used only once per K32 block. Force a
            # compiler-allocated VLM home rather than retaining 4*M total
            # vectors alongside 4*M partial vectors. No fixed VLM addresses:
            # the allocator accounts for this array and other spills together.
            for r in range(m):
                body=body.replace(f'float128 a{r}={{0}},t{r}={{0}},b{r}={{0}},u{r}={{0}};',
                    f'float128 a{r}={{0}},b{r}={{0}};'+''.join(f'totals[{4*r+j}]=(float64)0;'for j in range(4)))
                needle=f't{r}.v1=v_f32_mac_b'
                assert body.count(needle)==1
                body=body.replace(needle,f'float128 t{r}={{totals[{4*r}],totals[{4*r+1}]}},u{r}={{totals[{4*r+2}],totals[{4*r+3}]}};'+needle)
                body=body.replace(f'b{r}=(float128){{0}};',f'b{r}=(float128){{0}};totals[{4*r}]=t{r}.v1;totals[{4*r+1}]=t{r}.v2;totals[{4*r+2}]=u{r}.v1;totals[{4*r+3}]=u{r}.v2;')
                body=body.replace(f'linear_acc(t{r})',f'linear_acc((float128){{totals[{4*r}],totals[{4*r+1}]}})').replace(f'linear_acc(u{r})',f'linear_acc((float128){{totals[{4*r+2}],totals[{4*r+3}]}})')
        # All K/N regions are disjoint. No preprocess, shadow owner, decode
        # tensor or duplicated packed/scale allocation is introduced.
        body=body.replace('int5 p={0,g*32+(base+',f'int5 p={{(nb&1)*128,(source*{n//512}+nb/2)*K+g*32+(base+')
        body=body.replace('),nb,source,0};','),0,0,0};')
        assert 'int5 p={0,' not in body
        body=body.replace('int5 sp={0,g,nb,source,0};',f'int5 sp={{(nb&1)*256,(source*{n//512}+nb/2)*(K/32)+g,0,0,0}};')
        body=body.replace('sp[0]=128;', 'sp[0]+=128;')
        if not down or dense:
            for r in range(m):
                source+=f'int5 rm{r}={{{r},expert,0,0,0}};int route{r}=s_i32_ld_g(gen_addr(rm{r},row_map));int token{r}=route{r}'+(';'if down else'>>3;')+'\n'
                body=body.replace(f'int5 xp{r}={{g*32,{r},expert,0,0}};',f'int5 xp{r}={{g*32,token{r},0,0,0}};')
                if dense:body=body.replace(f'int5 o={{nb*256,{r},split,expert,0}};',f'int5 o={{nb*256,split,route{r},0,0}};')
        source+=body+'\n}\n'
    if wide_m1:source+='}}'
    return source+'}}}\n'
