#!/usr/bin/env python3
"""Seal the fixed capacity8 preflight from local payloads and remote SHA records.

No device APIs. The .pt files are our own captured tensors, so protocol-4
torch.load(weights_only=False) is intentional; never use this on untrusted data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Canonical repository containing downloaded assets')
    parser.add_argument('--case', default='production-capacity8-preflight-e',
                        help='Completed two-layer case with the same capacity8/4216 boundary fixture')
    parser.add_argument('--remote-hashes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Evidence output directory')
    args = parser.parse_args()
    name = args.case
    assert name and Path(name).name == name
    case = args.repo / 'artifacts/builds/production-integration/model-runs' / name
    runner = case.with_name(name + '-runner')
    remote = json.loads(args.remote_hashes.read_text())
    local_files = [dict(path=str(p.relative_to(case)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(case.rglob('*')) if p.is_file()]
    companions = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
                  for p in sorted(runner.iterdir()) if p.is_file()]
    assert local_files == remote['files'], 'remote/local case file set or bytes mismatch'
    assert companions == remote['runner_files'], 'remote/local runner file set or bytes mismatch'
    frozen = json.loads((case / 'source/sha256.json').read_text())
    for path, expected in frozen.items():
        assert sha(case / 'source' / path) == expected, ('frozen source mismatch', path)
    runner_exit = json.loads((runner / (name + '.exit.json')).read_text())
    assert json.loads((case / 'source/git-revision.json').read_text())['commit'] == runner_exit['source_commit']
    assert runner_exit['returncode'] == 0 and not runner_exit['remaining_owned_processes']
    assert runner_exit['preflight']['returncode'] == runner_exit['postflight']['returncode'] == 0
    result = json.loads((case / 'result.json').read_text())
    assert result['status'] == 'DIAGNOSTIC' and result['candidate_accepted'] is False
    assert result['config']['max_num_seqs'] == 8 and result['config']['hf_overrides']['num_hidden_layers'] == 2
    policies = list(dict.fromkeys(r['policy'] for r in result['runs']))
    assert len(policies) == 3 and policies[0] == 'bf16_fp32' and len(result['runs']) == 6
    positions = {4222, 4223, 4224, 4225}
    checks, comparisons, main_capture_names = [], [], set()
    for policy in policies:
        bridge = next(r for r in result['runs'] if r['policy'] == policy and not r['active'])
        native = next(r for r in result['runs'] if r['policy'] == policy and r['active'])
        assert len(bridge['ids']) == len(native['ids']) == 16 and bridge['ids'] == native['ids']
        assert len(bridge['ranks']) == len(native['ranks']) == 8
        assert bridge['performance_qualified'] is False and native['performance_qualified'] is False
        expected_padded = policy != 'bf16_fp32'
        rank_checks = []
        for rank, (left, right) in enumerate(zip(bridge['ranks'], native['ranks'])):
            for number, capture in enumerate(right['captures']):
                assert capture['plan_info'][3:] == [11, 6]
                main_capture_names.add(f'native-phase{right["phase"]}-capture{number}-rank{rank}.json')
            a = {v['position']: v for v in left['boundary_audit']}
            b = {v['position']: v for v in right['boundary_audit']}
            assert set(a) == set(b) == positions
            assert all(c['padded_window'] == expected_padded and c['capacities'] == [56, 8]
                       and c['status'] == 'NUMERICAL_REPLAY_PASS' for c in right['captures'])
            executed = {v['position'] for v in right['native_steps']}
            assert (4223 in executed) == expected_padded
            assert right['fallbacks'].get('window_one_page', 0) == (0 if expected_padded else 1)
            for position in sorted(positions):
                records = (a[position], b[position])
                assert records[0]['mode'] == 'bridge'
                assert records[1]['mode'] == ('bridge' if position == 4223 and not expected_padded else 'native')
                payloads = []
                for record in records:
                    path = case / 'boundary-audit' / Path(record['path']).name
                    assert sha(path) == record['sha256'], ('payload file SHA', path)
                    payload = torch.load(path, map_location='cpu', weights_only=False)
                    assert set(payload) == {'hidden', 'selected', 'logits'}
                    for key, tensor in payload.items():
                        metadata = record['tensors'][key]
                        assert list(tensor.shape) == metadata['shape'] and str(tensor.dtype) == metadata['dtype']
                        digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                        assert digest == metadata['sha256'], (policy, rank, position, key, 'tensor SHA')
                    payloads.append(payload)
                for key in ('hidden', 'selected', 'logits'):
                    lhs, rhs = (p[key] for p in payloads)
                    equal = (lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
                             and torch.equal(lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8)))
                    assert equal, (policy, rank, position, key, 'byte mismatch')
                    comparisons.append(dict(policy=policy, rank=rank, position=position, tensor=key,
                                            shape=list(lhs.shape), dtype=str(lhs.dtype), bytes=lhs.numel() * lhs.element_size(),
                                            bridge_sha256=records[0]['tensors'][key]['sha256'],
                                            candidate_sha256=records[1]['tensors'][key]['sha256'],
                                            candidate_mode=records[1]['mode'], all_bits_equal=equal))
            rank_checks.append(dict(rank=rank, padded=expected_padded, capacities=[56, 8],
                                    fallbacks=right['fallbacks'], native_boundary_positions=sorted(executed & positions),
                                    complete_tensor_bits_equal=True))
        checks.append(dict(policy=policy, rank_checks=rank_checks, token_count_pass=True, bridge_native_ids_match=True))
    assert len(comparisons) == 288
    profile_capture_names = {f'native-phase{row["phase"]}-capture{number}-rank{rank}.json'
                             for rank, row in enumerate(result['device_profile']['ranks'])
                             for number, _ in enumerate(row['captures'])}
    assert len(main_capture_names) == 24 and len(profile_capture_names) == 8
    assert not main_capture_names & profile_capture_names
    captures = []
    for path in sorted(case.glob('native-phase*-capture*-rank*.json')):
        capture = json.loads(path.read_text())
        header = path.with_name(path.stem + '-commands.txt').read_text().splitlines()[0]
        count = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', header)}
        assert capture['status'] == 'NUMERICAL_REPLAY_PASS'
        assert path.name in main_capture_names | profile_capture_names
        scope = 'main_preflight' if path.name in main_capture_names else 'separate_profile'
        expected_launches = 11 if scope == 'main_preflight' else 10
        assert capture['capacities'] == [56, 8] and capture['plan_info'][3:] == [expected_launches, 6]
        assert count['launches'] == expected_launches and count['collectives'] == 6
        captures.append(dict(path=path.name, scope=scope, capacities=capture['capacities'],
                             padded=capture['padded_window'], counts=count))
    assert len(captures) == 32
    launch_differences = []
    final_native = next(r for r in result['runs'] if r['policy'] == policies[-1] and r['active'])
    for rank, (main_rank, profile_rank) in enumerate(zip(final_native['ranks'], result['device_profile']['ranks'])):
        names = [f'native-phase{v["phase"]}-capture0-rank{rank}' for v in (main_rank, profile_rank)]
        launches = []
        for stem in names:
            rows = (case / (stem + '-commands.txt')).read_text().splitlines()
            launches.append([dict(re.findall(r'(\w+)=([^ ]+)', row)) for row in rows if re.match(r'^\d+ kind=1 ', row)])
        left, right = launches
        assert [(v['recipe_or_comm'], v['tensors']) for v in left[1:]] == [
            (v['recipe_or_comm'], v['tensors']) for v in right]
        owners = [dict(part.split('=', 1) for part in line.split())
                  for line in (case / (names[0] + '-storage.txt')).read_text().splitlines()
                  if line.startswith('call=1 ')]
        inputs = [v for v in owners if v['role'] == 'input']
        outputs = [v for v in owners if v['role'] == 'output']
        assert len(inputs) == 18 and len(outputs) == 9 and left[0]['tensors'] == '38'
        assert all(o['storage'] == inputs[2 * i + 1]['storage'] for i, o in enumerate(outputs))
        assert {v['shape'] for v in owners} == {'1,1,', '56,', '8,'}
        launch_differences.append(dict(rank=rank, main_commands=names[0] + '-commands.txt',
                                      profile_commands=names[1] + '-commands.txt',
                                      extra_first_launch=left[0], remaining_ten_recipe_handles_and_tensor_counts_equal=True,
                                      first_call_inputs=18, first_call_outputs=9,
                                      output_owners_are_second_input_of_each_pair=True,
                                      owners=owners))
    summary = dict(source_result_sha256=sha(case / 'result.json'), status=result['status'], candidate_accepted=False,
                   source_commit=runner_exit['source_commit'], init_s=result['init_s'], runner_elapsed_s=runner_exit['elapsed_s'],
                   context=4216, generated_tokens=16, max_num_seqs=8, layers=2, policies=checks,
                   checked_tensors=len(comparisons), all_boundary_tensor_hashes_verified=True,
                   independently_loaded_and_compared_tensor_bytes=True,
                   main_captured_plan_launches=11, main_capture_count=24,
                   separate_profile_launches=10, separate_profile_capture_count=8,
                   all_captured_plan_collectives=6, all_captured_capacities=[56, 8],
                   capture_count=len(captures), captures=captures,
                   separate_profile_launch_difference=launch_differences,
                   launch_difference_interpretation='The main capture adds a first recipe with only scalar/56/8 metadata owners and nine destination-alias outputs, consistent with graph-input copies. The remaining ten recipe handles and tensor counts are identical on every rank. The missing recipe has no retained ELF or profile events; exact physical nodes and cache-state cause are not established.',
                   quality_scope='two-layer interface diagnostic only; no serving throughput or full-model quality acceptance',
                   comparisons=comparisons, torch_cpu_version=torch.__version__)
    manifest = dict(root=str(case), remote=remote['remote'], files=local_files, file_count=len(local_files),
                    total_bytes=sum(v['bytes'] for v in local_files), all_remote_hashes_verified=True,
                    frozen_sources_verified=len(frozen), runner_root=str(runner), runner_files=companions,
                    runner_total_bytes=sum(v['bytes'] for v in companions), all_runner_remote_hashes_verified=True,
                    remote_hash_record_sha256=sha(args.remote_hashes), audit_script_sha256=sha(Path(__file__)))
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix, value in (('manifest', manifest), ('summary', summary)):
        path = args.out / (name + '-' + suffix + '.json')
        assert not path.exists(), ('refuse overwrite', path)
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(case_files=len(local_files), case_bytes=manifest['total_bytes'], runner_files=len(companions),
                          runner_bytes=manifest['runner_total_bytes'], frozen_sources=len(frozen), compared_tensors=len(comparisons),
                          captures=len(captures), status='PASS_OFFLINE_INDEPENDENT_SEAL'), indent=2))


if __name__ == '__main__':
    main()
