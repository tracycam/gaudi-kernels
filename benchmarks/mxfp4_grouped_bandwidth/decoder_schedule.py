"""Actual address reuse / next-window prefetch, not merely another unroll flag."""
def transform(source, schedule):
    assert schedule in ('increment','pipeline')
    start=source.index('        // Glue rejects K')
    end=source.index('\n      }\n    }\n}',start)
    s=['        int5 p={0,g*32,nb,source,0};',
       '        int5 o={nb*256,g*32,expert,0,0};']
    if schedule=='pipeline':
        for i in range(8):s.append(f'        ushort128 i{i}=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];')
    s.append('        for(int z=0;z<32;z+=8) {')
    if schedule=='increment':
        for i in range(8):s.append(f'          ushort128 i{i}=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];')
    for i in range(8):s.append(f'          bfloat256 q{i}=v_bf16_lookup_2c(i{i},0,SW_LUT_PTR,(bfloat256){{0}});')
    if schedule=='pipeline':
        for i in range(8):s.append(f'          i{i}=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16,(uchar256)0,z<24);++p[1];')
    for i in range(8):
        s += [f'          v_bf16_st_tnsr(o,output,scaled(q{i}.v1,s0));o[0]+=128;',
              f'          v_bf16_st_tnsr(o,output,scaled(q{i}.v2,s1));o[0]-=128;++o[1];']
    s.append('        }')
    return source[:start]+'\n'.join(s)+source[end:]
