"""Private N128/K-paired original-width TPC experiment, not a model policy.

One byte owns the same N lane at two adjacent K positions. A paired BV16
lookup therefore supplies two MACs into ONE accumulator instead of separate
N-half accumulators. All M rows share those decoded weights. No scalar
activation load is introduced. Each K32 block retains ordered FP32 MACs and
one FP32 scale FMA, as in the grouped comparator.
"""
from pathlib import Path


def body(rows, cap, window, indent="", tile=128):
    lines = [f"float128 a{r}={{0}},t{r}={{0}};" for r in range(rows)]
    lines += ["for(int g=split*chunk;g<(split+1)*chunk&&g<groups;++g){"]
    for r in range(rows):
        lines += [f"int5 xp{r}={{g*32,{r},expert,0,0}};",
                  f"uchar256 xv{r}=(uchar256)v_bf16_ld_tnsr_partial_b(xp{r},activation,31,0);",
                  f"xv{r}=v_u8_mov_dual_group_all_b(xv{r},-1,0,0,0,0,MkWrA(3,3,3,3),0);"]
    if window != 32:
        lines += ["#pragma loop_unroll(1)", f"for(int base=0;base<32;base+={window}){{"]
    # Expose four K positions at a time. Reordering different output rows is
    # legal; the two K values for any one row always accumulate in K order.
    for z in range(0, window, 4):
        zexpr = str(z) if window == 32 else f"base+{z}"
        lines += ["{", f"int5 p={{0,g*16+({zexpr})/2,nb,source,0}};",
                  "ushort128 i0=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];",
                  "ushort128 i1=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);",
                  "bfloat256 q0=v_bf16_lookup_2c(i0,0,SW_LUT_PTR,(bfloat256){0});",
                  "bfloat256 q1=v_bf16_lookup_2c(i1,0,SW_LUT_PTR,(bfloat256){0});"]
        for j in range(4):
            for r in range(rows):
                lines += ["{", f"uchar256 mask=(uchar256)v_u32_mov_b(0x81808180u+({zexpr}+{j})*0x02020202u);",
                          f"bfloat128 x=(bfloat128)v_u8_shuffle_b(xv{r},mask,0,0);",
                          f"a{r}=v_bf16_mac_acc32_b(q{j//2}.v{j%2+1},x,a{r},0,count>{r});", "}"]
        lines += ["}"]
    if window != 32:
        lines += ["}"]
    lines += ["int5 sp={0,g,nb,source,0};",
              "ushort128 s=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);",
              "float128 f=v_convert_bf16_to_f32_all_b((bfloat128)((s&255)<<7));"]
    for r in range(rows):
        lines += [f"t{r}.v1=v_f32_mac_b(a{r}.v1,f.v1,t{r}.v1);",
                  f"t{r}.v2=v_f32_mac_b(a{r}.v2,f.v2,t{r}.v2);a{r}=(float128){{0}};"]
    lines += ["}"]
    for r in range(cap):
        value = f"linear_acc(t{r})" if r < rows else "{0}"
        lines += ["{", f"float128 y={value};int5 o={{nb*128,{r},split,expert,0}};",
                  "v_f32_st_tnsr(o,output,y.v1);o[0]+=64;v_f32_st_tnsr(o,output,y.v2);", "}"]
    source = "\n".join(indent + line for line in lines)
    if tile == 256:
        for r in range(rows):
            source = source.replace(f"float128 a{r}={{0}},t{r}={{0}};", f"float128 a{r}={{0}},t{r}={{0}},b{r}={{0}},u{r}={{0}};")
        # Same-width N256 control layout, now with real count branches. Each
        # lookup supplies two N halves for one K. Share the activation shuffle.
        for z in range(0, window, 4):
            zexpr = str(z) if window == 32 else f"base+{z}"
            source = source.replace(f"g*16+({zexpr})/2", f"g*32+({zexpr})")
        source = source.replace('bfloat256 q0=v_bf16_lookup_2c',
            '++p[1];ushort128 i2=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);'
            '++p[1];ushort128 i3=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);\n'
            'bfloat256 q2=v_bf16_lookup_2c(i2,0,SW_LUT_PTR,(bfloat256){0});'
            'bfloat256 q3=v_bf16_lookup_2c(i3,0,SW_LUT_PTR,(bfloat256){0});\nbfloat256 q0=v_bf16_lookup_2c')
        # Replace entire operand forms once per K position, avoiding cascading
        # q0.v2 -> q1.v1 -> q2.v1 replacements.
        for r in range(rows):
            for j in range(4):
                source = source.replace(f"a{r}=v_bf16_mac_acc32_b(q{j//2}.v{j%2+1},x,a{r},0,count>{r});",
                    f"a{r}=v_bf16_mac_acc32_b(QTEMP{j}.v1,x,a{r},0,count>{r});"
                    f"b{r}=v_bf16_mac_acc32_b(QTEMP{j}.v2,x,b{r},0,count>{r});")
        source = source.replace('QTEMP', 'q')
        source = source.replace('float128 f=v_convert_bf16_to_f32_all_b',
            'sp[0]=128;ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);'
            'float128 f1=v_convert_bf16_to_f32_all_b((bfloat128)((s1&255)<<7));\n'
            'float128 f=v_convert_bf16_to_f32_all_b')
        for r in range(rows):
            source = source.replace(f'a{r}=(float128){{0}};',
                f'a{r}=(float128){{0}};u{r}.v1=v_f32_mac_b(b{r}.v1,f1.v1,u{r}.v1);'
                f'u{r}.v2=v_f32_mac_b(b{r}.v2,f1.v2,u{r}.v2);b{r}=(float128){{0}};')
        source = source.replace('nb*128', 'nb*256')
        for r in range(cap):
            value = f'linear_acc(u{r})' if r < rows else '(float128){0}'
            marker = f'int5 o={{nb*256,{r},split,expert,0}};\nv_f32_st_tnsr(o,output,y.v1);o[0]+=64;v_f32_st_tnsr(o,output,y.v2);'
            assert source.count(marker) == 1
            source = source.replace(marker, marker+f'o[0]+=64;y={value};v_f32_st_tnsr(o,output,y.v1);o[0]+=64;v_f32_st_tnsr(o,output,y.v2);')
    return source


def generate(root: Path, rows: int, window: int, branch: bool, tile=128):
    assert 1 <= rows <= 4 and window in (4, 8, 16, 32)
    s = f'''#define main unused_legacy_main
#include "{root / 'csrc/tpc/mxfp4_linear/gemv.c'}"
#undef main
void main(tensor packed,tensor scales,tensor table,tensor activation,tensor ids,tensor counts,tensor output){{
 set_lut_256(table);
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int K=get_dim_size(activation,0),groups=K/32;
 int splits=get_dim_size(output,2),chunk=(groups+splits-1)/splits;
 for(int split=begin[2];split<end[2];++split)
 for(int expert=begin[1];expert<end[1];++expert){{
  int5 ep={{expert,0,0,0,0}};
  int source=s_i32_ld_g(gen_addr(ep,ids)),count=s_i32_ld_g(gen_addr(ep,counts));
  for(int nb=begin[0];nb<end[0];++nb){{
'''
    if branch:
        for m in range(rows + 1):
            s += ("if" if m == 0 else "else if") + f"(count=={m}){{\n"
            s += body(m, rows, window, tile=tile) + "\n}\n"
    else:
        s += body(rows, rows, window, tile=tile) + "\n"
    return s + "}}}\n"
