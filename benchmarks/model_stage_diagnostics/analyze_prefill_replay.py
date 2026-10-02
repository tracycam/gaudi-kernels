"""Audit sealed v/w/x diagnostics; never turn truncated evidence into model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root):
    reports = []
    libraries = None
    for variant, suffix in [('legacy', 'v'), ('capture', 'w'), ('replay', 'x')]:
        name = f'production-prefill-graph-{variant}-2-{suffix}'
        case = root/name
        data = json.loads((case/'result.json').read_text())
        manifest = json.loads((case/'source/sha256.json').read_text())
        current = {k: v for k, v in manifest.items() if k.endswith('.so')}
        assert current and all(sha(case/'source'/k) == v for k, v in current.items())
        if libraries is None:
            libraries = current
        assert current == libraries, 'Math libraries changed across probes'
        assert data['candidate_accepted'] is False
        runner = root/(name+'-runner')
        exit_record = json.loads((runner/(name+'.exit.json')).read_text())
        assert exit_record['returncode'] == 0 and exit_record['cleanup']['reaped']
        assert not exit_record.get('remaining_owned_processes')
        runs = []
        previous_windows = {rank: [] for rank in range(8)}
        for run in data['runs']:
            ranks = sorted(run['production_state'], key=lambda x: x['rank'])
            assert [r['rank'] for r in ranks] == list(range(8))
            evidence = []
            for state in ranks:
                graph = state['prefill_graph']
                row = dict(rank=state['rank'], decisions=graph['rows'],
                           route_capture_counts=state['moe_route']['python_capture_calls'])
                if variant == 'replay':
                    execution = graph['execution']
                    assert execution['sdk_source']['sha256'] == 'a43b8a28f1384cbb0de248e538595c1db31088d25100f2475eddfb04dfad77e8'
                    windows = execution['windows']
                    old = previous_windows[state['rank']]
                    assert windows[:len(old)] == old
                    added = windows[len(old):]
                    assert len(added) == 2
                    for window in added:
                        assert window['completed'] and not window['nested_prompt_scope']
                        assert len(window['graph_decisions']) == 1
                        assert window['graph_decisions'][0]['graph_requested']
                        expected = int(run['active'])
                        assert window['cached_params_delta'] == dict(
                            cache_hits=expected, skip_replay_cnt=0, bypass_hpu_graphs=0, iteration_cnt=1)
                        assert window['replay_api']['replay'] == dict(successful_calls=expected, failed_calls=0)
                        for api, values in window['replay_api'].items():
                            if api != 'replay':
                                assert values == dict(successful_calls=0, failed_calls=0)
                    row['new_prompt_windows'] = added
                    previous_windows[state['rank']] = windows
                evidence.append(row)
            runs.append(dict(policy=run['policy'], native=run['active'],
                first_step_ms=run['steps'][0]['ms'], token_count_pass=run['token_count_pass'],
                same_policy_bridge_native_tokens_match=run['bitwise_token_match'],
                capture_gates_pass=run['all_capture_gates_pass'], ranks=evidence))
        placement = runner/'route-placement.json'
        placement_summary = None
        if placement.exists():
            raw = json.loads(placement.read_text())
            placement_summary = dict(path=str(placement), sha256=sha(placement),
                records=[{k: record[k] for k in ('graph', 'decoder_nodes', 'compute_nodes',
                    'workspace_bytes', 'decoded_weight_SRAM_address_union_bytes',
                    'all_SRAM_address_union_bytes', 'original_owners') if k in record}
                    for record in raw['records']],
                scope='Only the retained graph; shared dump filename does not establish rank coverage')
        reports.append(dict(case=name, result_sha256=sha(case/'result.json'),
            source_manifest_sha256=sha(case/'source/sha256.json'), init_s=data['init_s'],
            status=data['status'], candidate_accepted=False, runs=runs,
            fp32_quality={k: v for k, v in data['fp32_quality'].items() if k != 'audits'},
            staged_audits=data['fp32_quality']['audits'],
            placement=placement_summary))
    return dict(scope='Two-layer diagnostic only; first-step timing is one observation, not isolated prefill or qualified model TPS. Replay API return is not device completion or one Synapse launch.',
                frozen_math_libraries=libraries, cases=reports,
                all_rank_replay_pattern_verified=True, model_quality_qualified=False,
                performance_qualified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(result['cases']), libraries=len(result['frozen_math_libraries']),
                         all_rank_replay_pattern_verified=result['all_rank_replay_pattern_verified'])))


if __name__ == '__main__':
    main()
