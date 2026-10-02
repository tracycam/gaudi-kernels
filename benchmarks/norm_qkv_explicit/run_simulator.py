"""Compare archived norm+block quant ELFs with archived fused ELF, offline only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--norm-library', type=Path, required=True)
    p.add_argument('--block-library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-library', type=Path)
    p.add_argument('--fixture', action='append', choices=('random', 'zero', 'cancellation', 'odd_tail', 'outlier'))
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = root/'benchmarks/norm_qkv_explicit/simulator.cpp'
    binary = out/'simulator'
    build = ['g++', '-O2', '-std=c++17', '-I/usr/lib/habanatools/include', str(source),
             '-L/usr/lib/habanatools', '-ltpc_tests_core_ext', '-ldl',
             '-Wl,-rpath,/usr/lib/habanatools', '-o', str(binary)]
    with (out/'build.log').open('w') as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
    env = dict(os.environ, TPC_RUNNER='0', LD_LIBRARY_PATH='/usr/lib/habanatools')
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(280964)
    records = []
    fixtures=[('random', 6144), ('zero', 6144), ('cancellation', 6144), ('outlier', 6144) if a.candidate_library else ('odd_tail', 129)]
    modes=('norm','block','fused','grid24') if a.candidate_library else ('norm','block','fused')
    for name, width in fixtures:
        if a.fixture and name not in a.fixture:
            continue
        dest = out/name
        dest.mkdir()
        x = torch.randn(width, generator=generator).bfloat16()
        residual = torch.randn(width, generator=generator).bfloat16()
        gamma = (torch.randn(width, generator=generator)*.2+1).bfloat16()
        if name == 'zero':
            x.zero_(); residual.zero_()
        if name == 'cancellation':
            residual = -x
            residual[::7] += torch.tensor(.0078125, dtype=torch.bfloat16)
        if name == 'outlier':
            x[::128] *= 128
            gamma[1::7] *= .001
            gamma[2::11] *= -1
        for key, value in [('x', x), ('residual', residual), ('gamma', gamma)]:
            value.view(torch.int16).numpy().tofile(dest/(key+'.bin'))
        row = dict(name=name, M=1, H=width, modes={})
        records.append(row)
        for mode in modes:
            result = dest/mode
            result.mkdir()
            lib = a.block_library if mode == 'block' else a.candidate_library if mode == 'grid24' else a.norm_library
            inputs = dest/'norm' if mode == 'block' else dest
            command = [str(binary), str(lib.resolve()), mode, str(width), str(inputs), str(result)]
            with (result/'run.log').open('w') as log:
                try:
                    run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
                    code = run.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            detail = dict(command=command, returncode=code)
            row['modes'][mode] = detail
            if code == 0:
                detail.update(json.loads(re.findall(r'^\{.*\}$', (result/'run.log').read_text(), re.M)[-1]))
                with (result/'kernel.dis').open('w') as log:
                    subprocess.run(['tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2', '--no-show-raw-insn', str(result/'kernel.o')], stdout=log, check=True)
            (out/'result.json').write_text(json.dumps(dict(status='RUNNING', records=records), indent=2)+'\n')
            if code:
                raise RuntimeError(detail)
        norm = torch.from_numpy(np.fromfile(dest/'norm/norm.bin', dtype=np.uint16).copy().view(np.int16)).view(torch.bfloat16)
        r = (x + residual).bfloat16()
        rn = r.float()
        reference_norm = (rn*torch.rsqrt(rn.square().mean()+1e-6)*gamma.float()).bfloat16()
        padded = torch.nn.functional.pad(norm, (0, (-width)%128)).reshape(-1, 128).float()
        scale = padded.abs().amax(-1).clamp_min(1e-10)*torch.tensor(1/448, dtype=torch.float32)
        # Independent Torch CPU OCP RNE, followed by explicit native-half RNE.
        q = ((padded/scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
        q.view(torch.uint8).numpy().tofile(dest/'expected-q.bin')
        (scale*2).numpy().tofile(dest/'expected-scale.bin')
        reference_norm.view(torch.int16).numpy().tofile(dest/'cpu-fp32-norm.bin')
        r.view(torch.int16).numpy().tofile(dest/'expected-residual.bin')
        row['norm_cpu_fp32_diagnostic'] = dict(bf16_mismatches=int((norm.view(torch.int16)!=reference_norm.view(torch.int16)).sum()),
            relative_l2=float((norm.float()-reference_norm.float()).norm()/reference_norm.float().norm().clamp_min(1e-30)))
        row['mismatches'] = {}
        for mode in modes[1:]:
            row['mismatches'][mode] = dict(
                q=int(np.count_nonzero(np.fromfile(dest/mode/'q.bin', np.uint8)!=q.view(torch.uint8).numpy().reshape(-1))),
                scale=int(np.count_nonzero(np.fromfile(dest/mode/'scale.bin', np.uint32)!=(scale*2).numpy().view(np.uint32))))
        row['residual_mismatches'] = {mode:int(np.count_nonzero(np.fromfile(dest/mode/'residual.bin', np.uint16)!=r.view(torch.int16).numpy().view(np.uint16))) for mode in modes if mode!='block'}
        row['passed'] = not any(v for d in row['mismatches'].values() for v in d.values()) and not any(row['residual_mismatches'].values())
        print(json.dumps(row), flush=True)
        (out/'result.json').write_text(json.dumps(dict(status='RUNNING', records=records), indent=2)+'\n')
        if not row['passed']:
            raise RuntimeError('Actual ELF bytes/scales differed; raw outputs preserved')
    shutil.copy2(source, out/'simulator.cpp')
    shutil.copy2(__file__, out/'run_simulator.py')
    summary = dict(status='PASS_SIM_ONLY', device_verified=False, build_command=build, records=records,
        libraries={str(path.resolve()):sha(path) for path in (a.norm_library, a.block_library, a.candidate_library) if path},
        simulator_sha256=sha('/usr/lib/habanatools/libtpc_tests_core_ext.so'),
        scope='Actual archived ELF execution; simulator cycles are not device time or multicore latency')
    summary['files_sha256'] = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='result.json'}
    (out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
