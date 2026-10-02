"""Build N128 candidates and the SAME-SOURCE N256 control; never acquires HPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from generate import generate

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--rows', type=int, choices=(1, 2, 3, 4), required=True)
p.add_argument('--window', type=int, choices=(4, 8, 16, 32), default=32)
p.add_argument('--branch', action='store_true')
p.add_argument('--control', action='store_true')
p.add_argument('--n256', action='store_true', help='Use original N256 packing with generated count-specialized bodies')
a = p.parse_args()
if a.control and (a.rows == 3 or a.branch):
    p.error('N256 control supports original M1/2/4 only')
root = Path(__file__).resolve().parents[2]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('grouped_generate', root/'benchmarks/mxfp4_grouped_bandwidth/generate.py')
grouped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouped)
grouped.generate(root, out)
if not a.control:
    (out/'grouped.c').write_text(generate(root, a.rows, a.window, a.branch, tile=256 if a.n256 else 128))
(out/'ring_gate.c').write_bytes((root/'benchmarks/mxfp4_grouped_bandwidth/ring_gate.c').read_bytes())

# Keep fixtures, graph construction, independent oracle and timers identical.
probe = (root/'benchmarks/mxfp4_grouped_bandwidth/probe.cpp').read_text()
probe = probe.replace('"../../csrc/common/experiment_device.hpp"', '"'+str(root/'csrc/common/experiment_device.hpp')+'"')
probe = probe.replace('"manual_ring.hpp"', '"'+str(root/'benchmarks/mxfp4_grouped_bandwidth/manual_ring.hpp')+'"')
probe = grouped.replace_once(probe, '(M!=1&&M!=2&&M!=4)', '(M<1||M>4)')
probe = grouped.replace_once(probe, 'if(pattern=="mixed"&&M!=2)throw std::invalid_argument("mixed means alternating M1/M2");',
                            'if(mode!="tpc")throw std::invalid_argument("this experiment is TPC only");')
probe = grouped.replace_once(probe, 'pattern!="mixed"&&pattern!="cancel"', 'pattern!="mixed"&&pattern!="mixed3"&&pattern!="cancel"')
probe = grouped.replace_once(probe, 'pattern=="mixed"?1+e%2:M', 'pattern=="mixed"?1+e%M:pattern=="mixed3"?(e==0?2:1+e%3):M')
probe = grouped.replace_once(probe, 'if(mode!="tpc")', 'if(pattern=="mixed3"&&(M<3||E!=20))throw std::invalid_argument("mixed3 requires E20 cap>=3");if(mode!="tpc")')
if not (a.control or a.n256):
    probe = grouped.replace_once(probe, '{Tile/2,K,N/Tile,pool}', '{Tile,K/2,N/Tile,pool}')
    old = next(line for line in probe.splitlines() if 'for(int k=0;k<K;++k)for(int j=0;j<Tile/2;' in line)
    probe = grouped.replace_once(probe, old, '''  for(int k=0;k<K;k+=2)for(int j=0;j<Tile;++j){
   auto code=[&](int z){return mix(uint32_t((nb*Tile+j)*31337+(pattern=="cancel"?z/2:z)*97+e*131))&15;};
   wp[((uint64_t(e)*(N/Tile)+nb)*(K/2)+k/2)*Tile+j]=code(k)|(code(k+1)<<4);
  }''')
    probe = grouped.replace_once(probe,
        'wp[((uint64_t(src)*(N/Tile)+col/Tile)*K+k)*(Tile/2)+(col%Tile)/256*128+col%128],code=(byte>>((col%256>=128)*4))&15',
        'wp[((uint64_t(src)*(N/Tile)+col/Tile)*(K/2)+k/2)*Tile+col%Tile],code=(byte>>((k%2)*4))&15')
(out/'probe.cpp').write_text(probe)
commands = []
for name in ('grouped', 'decode', 'reduce', 'ring_gate'):
    defs = ([f'-DGK_SMALLM_ROWS={a.rows}', '-DGK_SMALLM_UNROLL=4', '-DGK_SMALLM_VECTOR=1',
             '-DGK_SMALLM_FOLD_SCALE=1', f'-DGK_SMALLM_STRAIGHT={a.window}'] if name == 'grouped' else
            ['-DGK_MXFP4_DECODE_UNROLL=4', '-DGK_MXFP4_DECODE_FAST_SCALE=2'] if name == 'decode' else [])
    for flag, suffix in (('-c', '.o'), ('-S', '.s')):
        commands.append(['tpc-clang', '-O2', '-ffp-contract=off', '-mcpu=gaudi2', *defs, flag, name+'.c', '-o', name+suffix])
    commands.append(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64', name+'.o', name+'_x86.o'])
    commands.append(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', name+'.o'])
tile = 256 if a.control or a.n256 else 128
commands += [
    ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-I/usr/include/habanalabs',
     f'-DGK_SMALLM_ROWS={a.rows}', f'-DGK_GROUPED_TILE={tile}', str(root/'benchmarks/mxfp4_grouped_bandwidth/glue.cpp'),
     'grouped_x86.o', 'decode_x86.o', 'reduce_x86.o', 'ring_gate_x86.o', '-o', 'libgrouped_tpc.so'],
    ['g++', '-O3', '-std=c++17', '-I/usr/include/habanalabs', f'-DGK_GROUPED_TILE={tile}', 'probe.cpp',
     '-L/usr/lib/habanalabs', '-Wl,-rpath,/usr/lib/habanalabs', '-lSynapse', '-o', 'probe']]
meta = dict(rows=a.rows, window=a.window, branch=a.branch, control=a.control, n256=a.n256,
            layout='N256' if tile == 256 else 'N128K2', commands=commands, device_verified=False, state='building')
try:
    with (out/'build.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command)+'\n')
            log.flush()
            if command[0] == 'tpc-llvm-objdump':
                with (out/(command[-1]+'.dis')).open('w') as dis:
                    subprocess.run(command, cwd=out, stdout=dis, stderr=log, check=True)
            else:
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    meta['state'] = 'built_not_device_verified'
finally:
    meta['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'build.json'}
    (out/'build.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(dict(state=meta['state'], output=str(out))))
