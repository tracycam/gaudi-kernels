"""Private grouped decoder specialization for its validated K % 32 == 0 domain.

Changes instruction scheduling only. Packed layout, scale application, expert
selection and per-element arithmetic are unchanged. Never use for ragged K.
"""

def specialize(source, window):
    assert window in (4,8,16,32)
    start=source.index('        int k=g*32;')
    end=source.index('\n      }\n    }\n}',start)
    lines=['        // Glue rejects K not divisible by 32; no scalar tail needed.',
           f'        for(int z=0;z<32;z+={window}) {{',
           '          int k=g*32+z;',
           '          int5 p={0,k,nb,source,0};']
    for i in range(window):
        lines.append(f'          ushort128 i{i}=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);'+('++p[1];' if i+1<window else ''))
    for i in range(window):
        lines.append(f'          bfloat256 q{i}=v_bf16_lookup_2c(i{i},0,SW_LUT_PTR,(bfloat256){{0}});')
    for i in range(window):
        lines.extend([f'          {{ int5 o={{nb*256,k+{i},expert,0,0}};',
                      f'            v_bf16_st_tnsr(o,output,scaled(q{i}.v1,s0));o[0]+=128;',
                      f'            v_bf16_st_tnsr(o,output,scaled(q{i}.v2,s1)); }}'])
    lines.append('        }')
    return source[:start]+'\n'.join(lines)+source[end:]
