"""Package exact offline ELFs plus schedule-preserving score debug clones."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--include', default='/usr/lib/habanatools/include')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out, build = a.output.resolve(), a.build.resolve()
out.mkdir(parents=True, exist_ok=False)
meta = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'commands': [], 'status': 'BUILDING', 'device_verified': False}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with (out / 'build.log').open('w') as log:
    def run(command, capture=None):
        meta['commands'].append(command)
        log.write(json.dumps(command) + '\n'); log.flush()
        if capture:
            with (out / capture).open('w') as target:
                subprocess.run(command, cwd=out, stdout=target, stderr=log, check=True)
        else:
            subprocess.run(command, cwd=out, stdout=log, stderr=log, check=True)
    try:
        for name, original in [('baseline', 'baseline'), ('quad', 'head_quad_select')]:
            shutil.copy2(build / (original + '.o'), out / (name + '.o'))
            run(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', name + '.o'], name + '.dis')
            debug = name + '_scores'
            run([sys.executable, str(root / 'benchmarks/swa128_ilp/debug_scores.py'), name + '.dis', '--variant', name, '--output', debug + '.s'])
            run(['tpc-clang', '-x', 'assembler', '-mcpu=gaudi2', '-c', debug + '.s', '-o', debug + '_slots.o'])
            # objcopy can rewrite its input without a distinct output argument.
            run(['objcopy', '--dump-section', '.text=' + debug + '.text', debug + '_slots.o', debug + '_extract_copy.o'])
            run(['objcopy', '--update-section', '.text=' + debug + '.text', name + '.o', debug + '.o'])
            run([sys.executable, str(root / 'benchmarks/swa128_ilp/fix_symbols.py'), debug + '.o', debug + '_slots.o'])
            run(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', debug + '.o'], debug + '.dis')
            assert sha(out / (name + '.o')) == sha(build / (original + '.o'))
        names = ['baseline', 'quad', 'baseline_scores', 'quad_scores']
        for name in names:
            run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64', name + '.o', name + '_x86.o'])
        run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-I' + a.include,
             str(root / 'csrc/host/swa128_ilp_glue.cpp'), *[name + '_x86.o' for name in names], '-o', 'libgaudi_swa128_ilp_tpc.so'])
        run(['g++', '-O2', '-std=c++17', '-I' + a.include, str(root / 'benchmarks/swa128_ilp/simulator_scores.cpp'),
             '-L/usr/lib/habanatools', '-ltpc_tests_core_ext', '-Wl,-rpath,/usr/lib/habanatools', '-o', 'simulator_scores'])
        meta['status'] = 'COMPILED'
    finally:
        meta['files_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        (out / 'build.json').write_text(json.dumps(meta, indent=2) + '\n')
