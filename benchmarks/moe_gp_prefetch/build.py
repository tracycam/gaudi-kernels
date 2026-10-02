"""Port qualified grouped prefetch ISA back to production token addressing.

Separate opt-in GUID and Torch namespace; no production library replacement.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('gp_prefetch_schedule', root/'benchmarks/mxfp4_grouped_bandwidth/prefetch.py')
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)
ref = root/'benchmarks/mxfp4_grouped_bandwidth/reference'
blob = (ref/'gp.o').read_bytes()
assert hashlib.sha256(blob).hexdigest() == 'fe7cd09c6bb2d20b02a6addc901916a44cc30a9cd03be3e9704116048ec72460'
assembly, plan = schedule.transform((ref/'gp.s').read_text(), True)
assert assembly.count('set_indx I6, b00010, S11') == 1
assembly = assembly.replace('set_indx I6, b00010, S11', 'set_indx I6, b00010, S13')
plan.update(grouped_activation_coordinate=False, activation_coordinate='original token S13',
            model_qualified=False, production_changed=False)
(out/'schedule.json').write_text(json.dumps(plan, indent=2)+'\n')
(out/'candidate.o').write_bytes(blob)
(out/'candidate.s').write_text(assembly)
for kind, file in [('host', 'glue.cpp'), ('torch', 'torch.cpp')]:
    source = (root/'csrc'/kind/('moe_gp_scale_tail_glue.cpp' if kind == 'host' else 'moe_gp_scale_tail.cpp')).read_text()
    source = source.replace('gk_moe_gp_scale_tail_v1', 'gk_moe_gp_prefetch_experiment')
    source = source.replace('gaudi_gp_scale_tail', 'gaudi_gp_prefetch')
    (out/file).write_text(source)
commands = [
    ['tpc-clang', '-mcpu=gaudi2', '-c', 'candidate.s', '-o', 'candidate_slots.o'],
    ['objcopy', '--dump-section', '.text=candidate.text', 'candidate_slots.o'],
    ['objcopy', '--update-section', '.text=candidate.text', 'candidate.o'],
    ['python3', str(root/'benchmarks/mxfp4_grouped_bandwidth/fix_literal_symbols.py'), 'candidate.o', 'candidate_slots.o'],
    ['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64', 'candidate.o', 'candidate_x86.o'],
    ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-I/usr/include/habanalabs',
     'glue.cpp', 'candidate_x86.o', '-o', 'libgaudi_gp_prefetch_tpc.so']]
meta = dict(state='building', commands=commands, source_elf_sha256=hashlib.sha256(blob).hexdigest(), device_verified=False)
try:
    with (out/'build.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command)+'\n'); log.flush()
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out/'candidate.dis').open('w') as f:
        subprocess.run(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', str(out/'candidate.o')], stdout=f, check=True)
    os.environ.setdefault('PT_HPU_LAZY_MODE', '1')
    os.environ.setdefault('MAX_JOBS', '2')
    import habana_frameworks.torch as ht
    from torch.utils.cpp_extension import load
    h = Path(ht.__file__).parent
    load(name='gaudi_gp_prefetch', sources=[str(out/'torch.cpp')], extra_include_paths=[str(h/'include')],
         extra_cflags=['-O2'], extra_ldflags=[f'-L{h}/lib', f'-Wl,-rpath,{h}/lib', '-lhabana_pytorch_plugin'],
         build_directory=str(out), is_python_module=False, verbose=True)
    meta['state'] = 'built_not_device_verified'
finally:
    meta['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'build.json'}
    (out/'build.json').write_text(json.dumps(meta, indent=2)+'\n')
