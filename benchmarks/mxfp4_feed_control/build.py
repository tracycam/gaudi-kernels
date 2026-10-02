"""Compile only; this script does not acquire a device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
src = Path(__file__).resolve().parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
commands = []
for name in ('bf16', 'fp8', 'bf16_fenced', 'fp8_fenced'):
    flags = ['-DFEED_FP8=1'] if name.startswith('fp8') else []
    if name.endswith('_fenced'):
        flags.append('-DFEED_FENCE=1')
    for flag, suffix in (('-c', '.o'), ('-S', '.s')):
        commands.append(['tpc-clang', '-O2', '-mcpu=gaudi2', *flags, flag,
                         str(src/'generate.c'), '-o', name+suffix])
    commands.append(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B',
                     'i386:x86-64', name+'.o', name+'_x86.o'])
    commands.append(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2',
                     '--no-show-raw-insn', name+'.o'])
commands.append(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs',
                 '-I/usr/include/habanalabs', str(src/'glue.cpp'),
                 'bf16_x86.o', 'fp8_x86.o', 'bf16_fenced_x86.o', 'fp8_fenced_x86.o', '-o', 'libfeed.so'])
commands.append(['g++', '-O3', '-std=c++17', '-I/usr/include/habanalabs',
                 str(src/'probe.cpp'), '-L/usr/lib/habanalabs',
                 '-Wl,-rpath,/usr/lib/habanalabs', '-lSynapse', '-o', 'feed'])
commands.append(['g++', '-O3', '-std=c++17', '-I/usr/include/habanalabs',
                 str(src/'contention.cpp'), '-L/usr/lib/habanalabs',
                 '-Wl,-rpath,/usr/lib/habanalabs', '-lSynapse', '-o', 'contention'])
record = dict(state='building', commands=commands, real_mxfp4=False)
try:
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            log.write(json.dumps(cmd)+'\n'); log.flush()
            if cmd[0] == 'tpc-llvm-objdump':
                with (out/(cmd[-1]+'.dis')).open('w') as dis:
                    subprocess.run(cmd, cwd=out, stdout=dis, stderr=log, check=True)
            else:
                subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    record['state'] = 'built'
finally:
    record['sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in out.iterdir() if f.is_file() and f.name != 'build.json'}
    (out/'build.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(state=record['state'], output=str(out))))
