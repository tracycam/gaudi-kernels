#!/usr/bin/env python3
"""Read sealed 70-b trace windows and extract embedded sum ELFs; no device APIs.

Requires pyelftools and tpc-llvm-objdump. Output must be a new directory.
The first fused node identifies a source-defined boundary, not a known ISA body.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

from elftools.elf.elffile import ELFFile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def windows(case, traces):
    records = []
    for rank_record in traces:
        rank = rank_record['rank']
        stack, intervals, compute_pids = {}, [], set()
        with (case / f'native-device-profile-rank{rank}.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                args, pid = event.get('args', {}), event['pid']
                op = args.get('op')
                if (event['ph'] == 'M' and event['name'] == 'process_name'
                        and args.get('name', '').startswith(('*TPC (', '*MME ('))):
                    compute_pids.add(pid)
                if pid not in compute_pids or not op or op == 'null' or event['ph'] not in ('B', 'E'):
                    continue
                key = (pid, event['tid'], event['name'])
                if event['ph'] == 'B':
                    assert key not in stack, ('nested event', rank, key)
                    stack[key] = event
                elif key in stack:
                    begin = stack.pop(key)
                    intervals.append(dict(start=begin['ts'], end=event['ts'], op=op,
                                          name=event['name'], recipe=args['recipe_id'], pid=pid))
        by_recipe = collections.defaultdict(list)
        for interval in intervals:
            by_recipe[interval['recipe']].append(interval)
        for token in rank_record['tokens']:
            token_windows = []
            for recipe in token['recipe_compute_spans']:
                lo, hi = recipe['start_us'], recipe['end_us']
                recipe_id = recipe['recipe_ids'][0]
                nodes = [v for v in by_recipe[recipe_id]
                         if v['start'] >= lo - .001 and v['end'] <= hi + .001]
                sums = [v for v in nodes if v['op'] in ('pf_sum_f32_f32', 'pf_sum_bf16_bf16')]
                if sums:
                    start, sum_end = min(v['start'] for v in sums), max(v['end'] for v in sums)
                    fused = [v for v in nodes if v['op'].startswith('fused_kernel') and v['start'] >= start]
                    if fused:
                        first_name = min(fused, key=lambda x: x['start'])['name']
                        first = [v for v in fused if v['name'] == first_name]
                        end = max(v['end'] for v in first)
                        assert min(v['start'] for v in first) >= sum_end - .001
                        kind = 'f32_sum_norm' if sums[0]['op'] == 'pf_sum_f32_f32' else 'bf16_sum_norm'
                        token_windows.append(dict(kind=kind, recipe=recipe_id, start=start, end=end,
                                                  sum_span=sum_end - start, fused_node=first[0]['op'],
                                                  fused_span=end - min(v['start'] for v in first)))
                first = [v for v in nodes if '/self_attn/attn/batch2block_matmul/' in v['name']]
                last = [v for v in nodes if '/self_attn/attn/block2batch_matmul/' in v['name']]
                if first and last:
                    token_windows.append(dict(kind='full_attention_mapping_span', recipe=recipe_id,
                                              start=min(v['start'] for v in first), end=max(v['end'] for v in last),
                                              first=min(first, key=lambda x: x['start'])['name'],
                                              last=max(last, key=lambda x: x['end'])['name']))
            token_windows.sort(key=lambda x: x['start'])
            assert all(x['end'] <= y['start'] for x, y in zip(token_windows, token_windows[1:])), (rank, 'overlap')
            groups = collections.defaultdict(list)
            for item in token_windows:
                groups[item['kind']].append(item)
            assert {key: len(value) for key, value in groups.items()} == {
                'f32_sum_norm': 68, 'bf16_sum_norm': 72, 'full_attention_mapping_span': 10}
            records.append(dict(rank=rank, token=token['token_index'], windows=token_windows,
                                groups={kind: dict(count=len(items),
                                                   sum_window_us=sum(w['end'] - w['start'] for w in items),
                                                   individual_median_us=statistics.median(w['end'] - w['start'] for w in items),
                                                   sum_actual_sum_node_us=sum(w.get('sum_span', 0) for w in items),
                                                   sum_fused_node_us=sum(w.get('fused_span', 0) for w in items))
                                        for kind, items in groups.items()}))
    assert len(records) == 24 and len({(v['rank'], v['token']) for v in records}) == 24
    summary = {}
    for kind in records[0]['groups']:
        rows = [r['groups'][kind] for r in records]
        summary[kind] = dict(counts=sorted({r['count'] for r in rows}), **{
            key: dict(min=min(r[key] for r in rows), median=statistics.median(r[key] for r in rows),
                      max=max(r[key] for r in rows))
            for key in ('sum_window_us', 'individual_median_us', 'sum_actual_sum_node_us', 'sum_fused_node_us')})
    return dict(scope='24 rank-token observations; windows geometrically disjoint; not guaranteed savings; dynamic fused bodies unknown',
                summary=summary, records=records)


def extract(library, out, objdump):
    metadata = {}
    with library.open('rb') as stream:
        library_elf = ELFFile(stream)
        symbols = {s.name: s for s in library_elf.get_section_by_name('.symtab').iter_symbols()}
        for name in ('sum_f32_f32', 'sum_bf16_bf16', 'sum_f32_bf16'):
            begin, end = (symbols[f'_binary_{name}_o_{which}'] for which in ('start', 'end'))
            section = library_elf.get_section(begin['st_shndx'])
            offset, size = begin['st_value'] - section['sh_addr'], end['st_value'] - begin['st_value']
            blob = section.data()[offset:offset + size]
            assert len(blob) == size and blob[:4] == b'\x7fELF'
            elf_path = out / f'{name}.o'
            elf_path.write_bytes(blob)
            embedded = ELFFile(io.BytesIO(blob))
            (out / f'{name}.c').write_bytes(embedded.get_section_by_name('.source').data())
            # The embedded TPC container reports ELF32/i386: never infer ISA
            # from that machine field or objdump silently prints x86 nonsense.
            command = [objdump, '-d', '--triple=tpc', '--mcpu=gaudi2', str(elf_path)]
            proc = subprocess.run(command, capture_output=True, text=True, check=True)
            assert 'TPC architecture: gaudi2' in proc.stdout
            (out / f'{name}.objdump').write_text(proc.stdout)
            metadata[name] = dict(elf_sha256=sha(elf_path), bytes=size, command=command,
                                  text_sha256=hashlib.sha256(embedded.get_section_by_name('.text').data()).hexdigest(),
                                  stderr=proc.stderr)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Canonical gaudi-kernels directory containing sealed assets')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--objdump', default='tpc-llvm-objdump')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.repo.resolve()
    case = base / 'artifacts/builds/production-integration/model-runs/production-isa-70-b'
    trace_path = base / 'evidence/production-integration/model-runs/production-isa-70-b-trace.json'
    report = windows(case, json.loads(trace_path.read_text()))
    write_json(args.out / 'windows.json', report)
    library = case / 'source/precision-source/precision-tpc/libprecision_tpc.so'
    elf_metadata = extract(library, args.out, args.objdump)
    input_paths = [trace_path, library, case / 'result.json', case / 'native_model_source.py']
    input_paths += sorted(case.glob('native-device-profile-rank*.jsonl'))
    input_paths += [case / 'source' / path for path in (
        'e1_profile.inc', 'precision_runtime.py', 'precision_ops.py', 'flow_b2b.py',
        'precision-source/precision_glue.cpp', 'precision-source/precision_ops.cpp',
        'production-runtime/plugin/vllm_gaudi/extension/ops.py',
        'production-runtime/vllm/vllm/model_executor/models/mimo_v2.py')]
    code_assets = sorted(str(p.relative_to(case)) for p in case.rglob('*')
                         if p.is_file() and p.suffix.lower() in ('.recipe', '.elf', '.hltv'))
    logs = {str(p.relative_to(case)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted((case / 'habana-logs').glob('*')) if p.is_file()
            and p.name in ('graph_compiler.log', 'graph_compiler_perf.log', 'fuser_lib_log.txt')}
    summary = dict(status='READ_ONLY_ANALYSIS_NO_DEVICE', windows=report['summary'],
                   window_scope=report['scope'], dynamic_code_asset_paths=code_assets,
                   compiler_logs=logs, extracted_sum_kernels=elf_metadata,
                   inputs={str(p.relative_to(base)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in input_paths},
                   script_sha256=sha(Path(__file__)))
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(dict(status=summary['status'], observations=len(report['records']), windows=report['summary']), indent=2))


if __name__ == '__main__':
    main()
