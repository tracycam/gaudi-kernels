"""Actual baseline/candidate ELF comparison. ISA simulator, never a card.

The address proof exhausts every decode position in [0,32767]. ELF runs cover
four numerical fixtures and selected cold/page/32K boundary positions, not all
32768 positions. Simulation cycle counts are not physical device timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/swa128_attention'))
from oracle import full_reference


def address_proof(width=2):
    tails = pairs = 0
    for index in range(128):
        matches = [(bank, lane) for bank in range(2) for lane in range(64)
                   if lane == index - bank * 64]
        assert matches == [(index // 64, index % 64)]
    for position in range(32768):
        logical0 = max(0, position - 127) // 128 * 128
        old, new = [], []
        for logical in (logical0, logical0 + 128):
            lo = min(128, max(0, position - 127 - logical))
            hi = max(0, min(128, position - logical + 1))
            old.extend(logical + t for t in range(lo, hi))
            t = lo
            while t + width - 1 < hi:
                assert 0 <= t < t + width - 1 < 128
                new.extend(logical + t + d for d in range(width))
                for d in range(width):
                    index = logical + t + d - (position - 127)
                    assert 0 <= index < 128
                t += width
                pairs += 1
            while t < hi:
                new.append(logical + t)
                t += 1
                tails += 1
        assert new == old == list(range(max(0, position - 127), position + 1))
    return dict(positions=32768, token_group_width=width, grouped_iterations=pairs,
                scalar_tails=tails, duplicate_or_missing_reads=0,
                scope='CPU integer address proof, not ELF executions or HBM transactions')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--candidates', nargs='+', default=['head_pair'])
    a = p.parse_args()
    build, out = a.build.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(281724)
    heads, slots = a.heads, 32768
    q = torch.randn(1, heads, 192).bfloat16()
    k = torch.randn(slots, 1, 192).bfloat16()
    v = torch.randn(slots, 1, 128).bfloat16()
    fixtures = [('first', 0, 'ordinary'), ('boundary', 128, 'ordinary'),
                ('offset64', 320, 'ordinary'), ('allmask', 128, 'allmask'),
                ('cold1', 1, 'ordinary'), ('cold63', 63, 'ordinary'),
                ('cold64', 64, 'ordinary'), ('page127', 127, 'ordinary'),
                ('page129', 129, 'ordinary'), ('last32766', 32766, 'ordinary'),
                ('last32767', 32767, 'ordinary'),
                ('cancel32704', 32704, 'cancellation'),
                ('sink32767', 32767, 'sink'),
                ('maskedgroup32767', 32767, 'maskedgroup')]
    report = {'status': 'RUNNING', 'device_verified': False,
              'scope': 'full attention ELF BF16 bit comparison; no physical timing',
              'heads': heads, 'slots': slots,
              'address_proof': {str(width): address_proof(width) for width in (2, 4)},
              'records': [], 'source_commit': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'elf_sha256': {name: hashlib.sha256((build / (name + '.o')).read_bytes()).hexdigest()
                             for name in ['baseline', *a.candidates]},
              'simulator_library_sha256': hashlib.sha256(Path(
                  '/usr/lib/habanatools/libtpc_tests_core_ext.so').read_bytes()).hexdigest()}
    shared = out / 'shared'
    shared.mkdir()
    for name, tensor in [('query', q), ('key', k), ('value', v)]:
        tensor.view(torch.int16).numpy().tofile(shared / (name + '.bin'))
    for name, position, kind in fixtures:
        dest = out / name
        dest.mkdir()
        pages = [255, 0]
        groups = [0, 0]
        sinks = torch.tensor([-torch.inf, .5]).repeat((heads + 1) // 2)[:heads].bfloat16()
        qc, kc = q, k
        if kind == 'allmask':
            pages = [-1, -1]
        elif kind == 'maskedgroup':
            groups = [1, 1]
        elif kind == 'sink':
            sinks.fill_(100)
        elif kind == 'cancellation':
            qc = q.clone()
            qc[..., 0::2] = 1
            qc[..., 1::2] = -1
            kc = k.clone()
            kc[..., 1::2] = kc[..., 0::2]
            kc[..., -1] += .0078125
        for field, tensor in [('query', qc), ('key', kc), ('value', v)]:
            target = dest / (field + '.bin')
            if tensor is {'query': q, 'key': k, 'value': v}[field]:
                os.link(shared / (field + '.bin'), target)
            else:
                tensor.view(torch.int16).numpy().tofile(target)
        sinks.view(torch.int16).numpy().tofile(dest / 'sinks.bin')
        for field, values in [('pages', pages), ('groups', groups), ('position', [position])]:
            np.array(values, dtype=np.int32).tofile(dest / (field + '.bin'))
        logical0 = max(0, position - 127) // 128 * 128
        effective_pages = [x if g == 0 else -1 for x, g in zip(pages, groups)]
        high, rounded = full_reference(qc, kc, v, effective_pages,
                                       [logical0, logical0 + 128], position, sinks)
        high.numpy().tofile(dest / 'fp64.bin')
        rounded.view(torch.int16).numpy().tofile(dest / 'reference_bf16.bin')
        record = {'case': name, 'position': position, 'pages': pages,
                  'groups': groups, 'runs': {}}
        report['records'].append(record)
        baseline = None
        for variant in ['baseline', *a.candidates]:
            command = [str(build / 'simulator'), str(heads), str(slots), str(dest),
                       str(dest / (variant + '.bin')), str(build / (variant + '.o'))]
            run = {'command': command, 'timeout_s': 45}
            record['runs'][variant] = run
            try:
                with (dest / (variant + '.log')).open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=log,
                                             env=dict(os.environ, TPC_RUNNER='0'), timeout=45)
                run['returncode'] = process.returncode
                if process.returncode:
                    raise RuntimeError('simulator returned failure')
                bits = np.fromfile(dest / (variant + '.bin'), dtype=np.uint16)
                assert len(bits) == heads * 128
                actual = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).double().reshape(high.shape)
                error = actual - high
                run.update(finite=bool(actual.isfinite().all()),
                           max_abs_fp64=float(error.abs().max()),
                           relative_l2_fp64=float(error.norm() / high.norm().clamp_min(1e-30)))
                if baseline is None:
                    baseline = bits
                run['baseline_bit_mismatches'] = int((bits != baseline).sum())
                raw = (dest / (variant + '.log')).read_text()
                run['simulated_execution'] = json.loads(re.findall(r'\{"cycles".*\}', raw)[-1])
                run['status'] = 'PASS' if run['finite'] and run['baseline_bit_mismatches'] == 0 else 'FAIL'
            except Exception as error:
                run.update(status='FAIL', error=repr(error))
            (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'case': name, 'variant': variant, **run}), flush=True)
        if any(run['status'] != 'PASS' for run in record['runs'].values()):
            break
    report['status'] = ('PASS_BITWISE' if len(report['records']) == len(fixtures)
                        and all(r['status'] == 'PASS' for c in report['records']
                                for r in c['runs'].values()) else 'FAILED')
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'PASS_BITWISE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
