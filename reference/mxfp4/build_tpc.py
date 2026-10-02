"""Preserve certified MAC/scale order; replace task/activation addressing only."""
from pathlib import Path
import hashlib
import json
import subprocess

r = Path(__file__).resolve().parent
b = r / 'tpc'
nop = 'nop; nop; nop; nop\n'
def scalar(s): return f'nop; {s}; nop; nop\n'
def load(s): return f'{s}; nop; nop; nop\n'
def store(s): return f'nop; nop; nop; {s}\n'
def run(args): subprocess.run(args, cwd=b, check=True)

original = (r / 'inputs/gemv.s').read_text()
template = (r / 'inputs/template.c').read_text()
for name, blocks, kp in [('direct_gp', 3, 2048), ('direct_down', 12, 256)]:
    s = original
    a = s.index('.LBB0_2:\n') + len('.LBB0_2:\n')
    z = s.index('mov.f32  V0,', a)
    # ids dim0 (top-k) lives at MMIO 0x150, checked by dim_probe.c.
    h = load('ld_l mmio S30, 0x150')
    h += scalar('mul.u32 upper32 S11, S32, 0xaaaaaaab') + nop * 5
    h += scalar(f'shr.u32 S11, S11, {1 if blocks==3 else 3}') + nop * 5
    h += scalar(f'mul.i32 S12, S11, {blocks}') + nop * 6
    h += scalar('sub.i32 S12, S32, S12')
    # floor(2^32/topk) reciprocal: q is exact or one too small.
    # Correct remainder >= divisor, then special-case topk=1 (reciprocal wraps).
    h += scalar('mul.u32 upper32 S13, S11, S31') + nop * 5
    h += scalar('mul.i32 S14, S13, S30') + nop * 5
    h += scalar('sub.i32 S14, S11, S14') + nop * 5
    h += scalar('cmp_geq.u32 SP2, S14, S30') + nop * 5
    h += scalar('add.i32 S13, S13, 1, SP2')
    h += scalar('cmp_eq.i32 SP2, S30, 1') + nop * 5
    h += scalar('mov.i32 S13, S11, SP2') + nop * 5
    h += scalar('mul.i32 S14, S13, S30') + nop * 6
    h += scalar('sub.i32 S14, S11, S14') + nop * 6
    h += load('set_indx I0, b11111, 0x0') + nop * 5
    h += scalar('set_indx I0, b00001, S14') + nop * 5
    h += scalar('set_indx I0, b00010, S13') + nop * 5
    h += store('gen_addr AD0, 0x4, I0') + nop * 6 + load('ld_g S10, AD0') + nop * 12
    h += scalar(f'mul.i32 S10, S10, {blocks}') + nop * 6
    h += scalar('add.i32 S10, S10, S12') + nop * 6
    h += scalar('mul.i32 S1, S10, S0') + scalar('mul.i32 S3, S10, S2')
    if name == 'direct_gp':
        h += scalar('shl.i32 S15, S12, 0x6')
    h += nop * 6
    s = s[:a] + h + s[z:]
    s = s.replace('main:\n', 'main:\n'+scalar('mov.i32 S31, S0'))
    assert s.count('ld_l mmio S0, 0xb0;') == 1
    s = s.replace('ld_l mmio S0, 0xb0;', f'mov.i32 S0, {kp};')
    assert s.count('set_indx I6, b00010, S32') == 1
    s = s.replace('set_indx I6, b00010, S32', 'set_indx I6, b00010, '+('S13' if name=='direct_gp' else 'S11'))
    if name == 'direct_gp':
        old = 'nop; shl.i32 S7, S33, 0x5; nop; nop\n'
        assert s.count(old) == 1
        s = s.replace(old, scalar('add.i32 S7, S33, S15') + nop * 4 + scalar('shl.i32 S7, S7, 0x5'))
    (b / (name+'.s')).write_text(s)
    t = template.replace('tensor mapping', 'tensor ids').replace('const int K=get_dim_size(activation,0);', f'const int K={kp};')
    t = t.replace('tensor output)', 'tensor output,int topk_magic)')
    old = 'int5 mi={block,0,0,0,0}; int mapped=s_i32_ld_g(gen_addr(mi,mapping));'
    assert old in t
    t = t.replace(old, f'int route=block/{blocks}, sub=block%{blocks}, topk=get_dim_size(ids,0); '
                  f'int5 mi={{route%topk,route/topk,0,0,0}}; '
                  f'int mapped=s_i32_ld_g(gen_addr(mi,ids))*{blocks}+sub;')
    t = t.replace('xc[0]=k;', 'xc[0]=sub*K+k;xc[1]=route/topk;' if name=='direct_gp' else 'xc[0]=k;xc[1]=route;')
    (b / (name+'.c')).write_text(t)
    run(['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c',name+'.c','-o',name+'.o'])
    run(['tpc-clang','-mcpu=gaudi2','-c',name+'.s','-o',name+'_slots.o'])
    run(['objcopy','--dump-section','.text='+name+'.text',name+'_slots.o'])
    run(['objcopy','--update-section','.text='+name+'.text',name+'.o'])
s=original
a=s.index('mov.f32  V0,',s.index('.LBB0_2:'))
s=s[:a]+scalar('cmp_geq.i32 SP2, S10, 0')+scalar('cmp_grt.i32 SP1, S0, 31')+nop*5+scalar('and.b SP1, SP1, SP2')+nop*5+s[a:]
(b/'masked_gemv.s').write_text(s)
(b/'masked_gemv.c').write_text(template)
run(['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c','masked_gemv.c','-o','masked_gemv.o'])
run(['tpc-clang','-mcpu=gaudi2','-c','masked_gemv.s','-o','masked_gemv_slots.o'])
run(['objcopy','--dump-section','.text=masked_gemv.text','masked_gemv_slots.o'])
run(['objcopy','--update-section','.text=masked_gemv.text','masked_gemv.o'])
for name in ['compact_gate','gate_broadcast','sorted_prep','sorted_combine']:
    run(['tpc-clang','-O2','-mcpu=gaudi2','-c',name+'.c','-o',name+'.o'])
    run(['tpc-clang','-O2','-mcpu=gaudi2','-S',name+'.c','-o',name+'.s'])
names = ['direct_gp','compact_gate','direct_down','gate_broadcast','sorted_prep','sorted_combine','masked_gemv']
for name in names:
    run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',
     str(r/'glue.cpp'),*[n+'_x86.o' for n in names],'-o','libbatch_tpc.so'])
(b/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in b.iterdir() if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
