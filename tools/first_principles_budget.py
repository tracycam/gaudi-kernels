"""Recompute a cost ledger from retained measurements; never acquire a device."""
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
timings_path = root / 'evidence/fp8-overhead/rows-main-f/summary.json'
timeline_path = root / 'evidence/fp8-overhead/profile-rows-f/timeline.json'
timings = json.loads(timings_path.read_text())
timeline = json.loads(timeline_path.read_text())['records']


def union(nodes):
    end = float('-inf')
    result = 0.
    for node in sorted(nodes, key=lambda n: n['start_us']):
        result += max(0., node['end_us'] - max(end, node['start_us']))
        end = max(end, node['end_us'])
    return result


m, n, k = 512, 1024, 2048
flops = 2*m*n*k
records = []
for mode, peak in [('w8a8', 865), ('w8a16', 432)]:
    measured = next(t for t in timings if (t['mode'], t['M'], t['N'], t['K']) == (mode,m,n,k))
    traced = next(t for t in timeline if t['case'] == f'{mode}-{m}-{n}-{k}-random')
    stages = {}
    predicates = {
        'quantization': lambda t: 'quantize' in t['name'],
        'decode': lambda t: 'decode' in t['name'],
        'mme': lambda t: t['unit'] == 'MME',
        'epilogue': lambda t: 'scale_bias' in t['name'],
    }
    for name, predicate in predicates.items():
        stages[name] = statistics.median(union([v for v in i['nodes'] if predicate(v)])
                                          for i in traced['invocations'])
    floor = flops/(peak*1e6)
    elapsed = measured['event_median_us']
    record = {'contract': mode, 'M': m, 'N': n, 'K': k, 'useful_flops': flops,
        'nominal_mme_peak_tflops': peak, 'nominal_compute_floor_us': floor,
        'measured_native_recipe_us': elapsed, 'measured_node_envelopes_us': stages,
        'recipe_useful_tflops': flops/elapsed/1e6,
        'recipe_nominal_peak_fraction': floor/elapsed,
        'mme_envelope_nominal_peak_fraction': floor/stages['mme'],
        'measured_tpc_mme_overlap_us': traced['median']['tpc_mme_overlap_us'],
        'logical_bytes': {'stored_weights': n*k, 'original_activations': 2*m*k,
                          'fp32_mme_output': 4*m*n, 'bf16_final_output': 2*m*n}}
    if mode == 'w8a8':
        assert stages['quantization'] > 0, 'quantizer node was not identified'
        record['logical_bytes'].update(quantizer_two_input_passes=4*m*k, quantized_activation=m*k,
                                       activation_scale=4*m)
        record['conditional_sensitivity_only'] = [
            {'assumed_quantization_speedup': speedup,
             'recipe_us_if_all_other_time_is_unchanged': elapsed-stages['quantization']*(1-1/speedup)}
            for speedup in [2,3,4]]
    else:
        record['logical_bytes']['decoded_sram_weights'] = 2*n*k
    records.append(record)

print(json.dumps({'status':'OFFLINE_RECOMPUTATION', 'records':records,
    'source_sha256': {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [timings_path,timeline_path]},
    'limitations': [
        'Stage times are unioned node envelopes, not physical ALU busy counters.',
        'Recipe times and profile stages come from distinct runs; residual is not attributed to CPU.',
        'Logical bytes are not measured HBM transactions; SRAM/cache and overlap matter.',
        'Conditional sensitivity is not new performance evidence or a prediction.',
        'The FP8 quantizer already uses one reciprocal per token, not per element.',
        'Nominal peak is a reference denominator; arbitrary shapes cannot attain it.'
    ]}, indent=2))
