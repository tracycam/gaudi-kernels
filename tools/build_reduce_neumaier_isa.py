"""Build separate ISA candidates without modifying any qualified production ELF."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--include', default='/usr/lib/habanatools/include')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
shutil.copy2(a.baseline, out / 'neumaier_baseline.o')
source_files = ['csrc/tpc/block_fp8_reduce_isa/lookahead.c',
                'csrc/tpc/block_fp8_reduce_experiment/reduce_neumaier.c',
                'csrc/host/block_fp8_reduce_isa_glue.cpp',
                'benchmarks/reduce_neumaier_isa/schedule.py',
                'benchmarks/reduce_neumaier_isa/simulator.cpp',
                'benchmarks/reduce_neumaier_isa/debug_fp32.py',
                'benchmarks/reduce_neumaier_isa/fix_symbols.py',
                'benchmarks/reduce_neumaier_isa/audit.py',
                'benchmarks/reduce_neumaier_isa/run_simulator.py',
                'tools/build_reduce_neumaier_isa.py']
for rel in source_files:
    dst = out / 'source' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, dst)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
commands = []
meta = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)),
            baseline_sha256=sha(a.baseline), source_hashes={rel: sha(root / rel) for rel in source_files},
            commands=commands, device_tested=False)
try:
    with (out / 'build.log').open('w') as log:
        def run(cmd, capture=None):
            commands.append(cmd)
            log.write(json.dumps(cmd) + '\n')
            log.flush()
            if capture:
                with (out / capture).open('w') as stdout:
                    subprocess.run(cmd, cwd=out, stdout=stdout, stderr=log, check=True)
            else:
                subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        def disasm(name):
            run(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', name + '.o'], name + '.dis')
        for flag, ext in [('-c', '.o'), ('-S', '.s')]:
            run(['tpc-clang', '-O2', '-ffp-contract=off', '-mcpu=gaudi2', flag,
                 str(out / 'source/csrc/tpc/block_fp8_reduce_isa/lookahead.c'), '-o', 'neumaier_lookahead' + ext])
        disasm('neumaier_baseline')
        disasm('neumaier_lookahead')
        run([sys.executable, str(out / 'source/benchmarks/reduce_neumaier_isa/schedule.py'),
             'neumaier_lookahead.dis', '--out', 'neumaier_handschedule.s'])
        run(['tpc-clang', '-x', 'assembler', '-mcpu=gaudi2', '-c', 'neumaier_handschedule.s', '-o', 'neumaier_handschedule_slots.o'])
        # Retain the compiler's tensor metadata and replace .text only in a new candidate.
        run(['objcopy', '--dump-section', '.text=neumaier_handschedule.text', 'neumaier_handschedule_slots.o'])
        shutil.copy2(out / 'neumaier_lookahead.o', out / 'neumaier_handschedule.o')
        run(['objcopy', '--update-section', '.text=neumaier_handschedule.text', 'neumaier_handschedule.o'])
        run([sys.executable, str(out / 'source/benchmarks/reduce_neumaier_isa/fix_symbols.py'), 'neumaier_handschedule.o', 'neumaier_handschedule_slots.o'])
        disasm('neumaier_handschedule')
        names = ['neumaier_baseline', 'neumaier_lookahead', 'neumaier_handschedule']
        for name in names[:]:
            source = name + ('.s' if name == 'neumaier_handschedule' else '.dis')
            debug = name + '_debug'
            run([sys.executable, str(out / 'source/benchmarks/reduce_neumaier_isa/debug_fp32.py'), source, '--out', debug + '.s'])
            run(['tpc-clang', '-x', 'assembler', '-mcpu=gaudi2', '-c', debug + '.s', '-o', debug + '_slots.o'])
            run(['objcopy', '--dump-section', '.text=' + debug + '.text', debug + '_slots.o'])
            shutil.copy2(out / (name + '.o'), out / (debug + '.o'))
            run(['objcopy', '--update-section', '.text=' + debug + '.text', debug + '.o'])
            run([sys.executable, str(out / 'source/benchmarks/reduce_neumaier_isa/fix_symbols.py'), debug + '.o', debug + '_slots.o'])
            disasm(debug)
            names.append(debug)
        for name in names:
            run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64', name + '.o', name + '_x86.o'])
        run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-I' + a.include,
             str(out / 'source/csrc/host/block_fp8_reduce_isa_glue.cpp'),
             *[name + '_x86.o' for name in names], '-o', 'libreduce_neumaier_isa.so'])
        run(['g++', '-O2', '-std=c++17', '-I' + a.include,
             str(out / 'source/benchmarks/reduce_neumaier_isa/simulator.cpp'),
             '-L' + str(out), '-lreduce_neumaier_isa', '-L/usr/lib/habanatools', '-ltpc_tests_core_ext',
             '-Wl,-rpath,' + str(out) + ':/usr/lib/habanatools', '-o', 'simulator'])
        run([sys.executable, str(out / 'source/benchmarks/reduce_neumaier_isa/audit.py'), '--build', str(out), '--out', 'isa-audit.json'])
    meta['compiled'] = True
except Exception as e:
    meta.update(compiled=False, error=str(e))
    raise
finally:
    meta['artifact_hashes'] = {path.name: sha(path) for path in out.iterdir() if path.is_file() and path.name != 'build.json'}
    (out / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
