"""Re-audit the retrieved native smoke and keep offline GP proof separate."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
case = args.artifacts / 'grouped-e2t2-n512k6144-v1'
status = json.loads((case / 'exit.json').read_text())
fixture = json.loads((case / 'fixture.json').read_text())
graph = json.loads((case / 'post_graph.json').read_text())['graphs'][0]
output = np.fromfile(case / 'output.bin', np.float32).astype(np.float64)
ref = np.fromfile(case / 'reference.f64', np.float64)
norm = np.fromfile(case / 'sumabs.f64', np.float64)
error = np.abs(output - ref)
bad = int(np.count_nonzero(~np.isfinite(output) | (error > 2e-6 * norm + 1e-35)))
assert output.size == fixture['T'] * fixture['N'] and not bad
assert not np.any(np.fromfile(case / 'overflow.bin', np.int32))
assert status['runner_exit_code'] == 0 and status['source_identity_verified']
nodes = [n for n in graph['nodes'] if not n['is_logical']]
assert [n['engine'] for n in nodes] == ['TPC', 'TPC', 'MME', 'TPC']
assert nodes[2]['guid'] == 'batch_gemm'
decoded = next(t for t in graph['tensors'] if t['name'].endswith('_decoded_sram'))
assert decoded['allocation'] == 'SRAM' and decoded['rmw_section'] and not decoded['persistent']
decoded_bytes = math.prod(decoded['max_shape']) * decoded['dtype_bit_size'] // 8
assert decoded_bytes == 12 * 1024**2
timings = [json.loads(line) for line in (case / 'run.log').read_text().splitlines()
           if line.startswith('{') and json.loads(line).get('stage') == 'timing']
offline = []
for name in ['capT', 'overflow']:
    rows = [json.loads(line) for line in (args.artifacts / 'compile-v4' / f'sim-{name}.log').read_text().splitlines()
            if line.startswith('{')]
    assert len(rows) == 1 and rows[0]['bad'] == 0 and not rows[0]['device_validated']
    offline += rows
summary = {
    'all_executed_grouped_checks_pass': True,
    'production_qualified': False,
    'native_linear_smoke': {
        'commit': status['git_commit'], 'module': 7, 'case': case.name,
        'shape': {k: fixture[k] for k in ['N', 'K', 'T', 'R', 'E', 'M_e']},
        'checked': int(output.size), 'bad': bad,
        'max_abs': float(error.max()), 'max_backward': float((error / norm).max()),
        'event_us_median': statistics.median(t['event_us'] for t in timings),
        'wall_us_median': statistics.median(t['wall_us'] for t in timings),
        'all_weight_scale_bytes': fixture['E'] * fixture['N'] * fixture['K'] * 17 // 32,
        'active_weight_scale_bytes': sum(m > 0 for m in fixture['M_e']) * fixture['N'] * fixture['K'] * 17 // 32,
        'decoded_sram_bytes': decoded_bytes, 'decoded_allocation': decoded['allocation'],
        'compute_guids': [n['guid'] for n in nodes], 'workspace_bytes': graph['workspace_size'],
        'kernel_sha256': next(iter(status['kernel_libraries_sha256'].values())),
        'physical_HBM_counters_measured': False,
        'placement': [{k: t[k] for k in ['name', 'allocation', 'dtype', 'max_shape', 'persistent', 'rmw_section']}
                      for t in graph['tensors']],
    },
    'offline_GP_extension': {
        'commit': json.loads((args.artifacts / 'compile-v4' / 'source-identity.json').read_text())['git_commit'],
        'kernel_sha256': hashlib.sha256((args.artifacts / 'compile-v4' / 'libmxfp4_moe_grouped_tpc.so').read_bytes()).hexdigest(),
        'simulator_checks': sum(r['checked'] for r in offline), 'cases': offline,
        'full_GP_recipe_device_validated': False,
        'native_probe_compiles_only': True,
        'fixture_shapes_ready': ['E2/T1/R2/N512/K6144', 'E2/T2/R2/N512/K6144', 'E2/T8/R1/N512/K6144 hot expert, one empty'],
    },
    'limits': ['No model installation or TPS claim', 'CAP<T overflow is unsupported diagnostic, never silently dropped',
               'Static E*CAP MME slots still execute for empty experts', 'Bucket routing is CPU capacity proof only',
               'Fast arithmetic certificate is separate from E8M0 scale eligibility',
               'Native-v2 aligned N512/K32 only; M1 production remains literal legacy'],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'native_checked': output.size, 'offline_checked': summary['offline_GP_extension']['simulator_checks'],
                  'production_qualified': False}, default=int))
