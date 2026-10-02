"""Compile and simulate a post-TopK candidate; deliberately retain tiny failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
c = root/'csrc/tpc/moe_router/post_top8.c'
cpp = Path(__file__).with_name('simulate_post.cpp')
commands = [['/usr/bin/tpc-clang', '-O2', '-ffp-contract=off', '-mcpu=gaudi2', mode, str(c), '-o', str(out/('post_top8'+suffix))]
            for mode, suffix in (('-c', '.o'), ('-S', '.s'))]
commands.append(['g++', '-O2', '-std=c++17', '-I/usr/lib/habanatools/include', str(cpp),
                 '-L/usr/lib/habanatools', '-ltpc_tests_core_ext', '-Wl,-rpath,/usr/lib/habanatools', '-o', str(out/'simulator')])
result = {'status': 'STARTED', 'TPC_RUNNER': '0', 'process_timeout_seconds': 45,
          'device_runtime_verified': False, 'commands': commands, 'cases': [],
          'source_sha256': {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in (c, cpp, Path(__file__))}}
result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
def save(): (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
save()
try:
    with (out/'compile.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command)+'\n'); log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=45)
    rng = np.random.default_rng(927384)
    ids = np.array([383, 0, 255, 128, 63, 64, 129, 382], np.int32)
    cases = [('normal', rng.uniform(.001, 1, 384).astype(np.float32), 2.5, 1),
             ('ordered_ties', np.full(384, .5, np.float32), 2.5, 1),
             ('near_ulp', np.resize(np.array([.5, np.nextafter(np.float32(.5), np.float32(1)),
                                            np.nextafter(np.float32(.5), np.float32(0))], np.float32), 384), 1., 1),
             ('no_renorm', rng.uniform(.001, 1, 384).astype(np.float32), 2.5, 0),
             ('small_normal', rng.uniform(1e-20, 1e-19, 384).astype(np.float32), 1., 1),
             ('zero', np.zeros(384, np.float32), 2.5, 1),
             ('subnormal', np.full(384, np.float32(1e-40)), 1., 1)]
    for name, scores, factor, renorm in cases:
        dest = out/name; dest.mkdir()
        scores.tofile(dest/'scores.bin'); ids.tofile(dest/'ids.bin')
        command = [str(out/'simulator'), str(out/'post_top8.o'), str(dest/'scores.bin'), str(dest/'ids.bin'),
                   str(dest/'weights.bin'), str(dest/'output-ids.bin'), str(dest/'sum.bin'), str(factor), str(renorm)]
        with (dest/'simulation.log').open('w') as log:
            subprocess.run(command, env=dict(os.environ, TPC_RUNNER='0'), cwd=dest,
                           stdout=log, stderr=subprocess.STDOUT, timeout=45, check=True)
        weights = np.fromfile(dest/'weights.bin', np.float32)
        output_ids = np.fromfile(dest/'output-ids.bin', np.int32)
        actual_sum = np.fromfile(dest/'sum.bin', np.float32)[0]
        selected = scores[ids]
        with np.errstate(all='ignore'):
            ideal = selected.astype(np.float64)
            if renorm: ideal = ideal / ideal.sum()
            ideal *= factor
            # Localize division correction independently from reduction order.
            once = (selected.astype(np.float64)/np.float64(actual_sum)).astype(np.float32) if renorm else selected
            once = (once*np.float32(factor)).astype(np.float32) if factor != 1 else once
        ideal.tofile(dest/'fp64-reference.bin'); once.tofile(dest/'same-denominator-fp32-reference.bin')
        finite = np.isfinite(ideal)
        maxerr = (float(np.max(np.abs(weights.astype(np.float64)-ideal)[finite]))
                  if finite.any() and np.isfinite(weights[finite]).all() else None)
        equal = np.array_equal(weights.view(np.uint32), once.view(np.uint32)) or (np.isnan(weights).all() and np.isnan(once).all())
        row = {'name': name, 'command': command, 'factor': factor, 'renormalize': bool(renorm),
               'ordered_ids_exact': bool(np.array_equal(ids, output_ids)), 'actual_sum': float(actual_sum),
               'fp64_sum': float(selected.astype(np.float64).sum()),
               'same_denominator_fp32_equal': bool(equal), 'max_abs_vs_fp64': maxerr,
               'finite_expected': bool(finite.all()), 'finite_actual': bool(np.isfinite(weights).all()),
               'weights': [float(v) if np.isfinite(v) else str(v) for v in weights]}
        result['cases'].append(row); save()
    mismatch = any(c['finite_expected'] and not c['finite_actual'] for c in result['cases'])
    result['status'] = 'ISA_SIMULATED_SUBNORMAL_FP64_MISMATCH' if mismatch else 'ISA_SIMULATED_NOT_DEVICE_QUALIFIED'
    result['all_ordered_ids_exact'] = all(c['ordered_ids_exact'] for c in result['cases'])
except Exception as error:
    result.update(status='FAIL', error=repr(error)); raise
finally:
    save()
print(json.dumps(result, indent=2))
raise SystemExit(1 if result['status'] == 'ISA_SIMULATED_SUBNORMAL_FP64_MISMATCH' else 0)
