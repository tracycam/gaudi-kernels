"""Summarize actual model evidence without promoting diagnostic performance."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--executor', type=Path, help='Independently rerun the unchanged quality classifier without changing the source record')
a = p.parse_args()
source = a.case/'result.json'
r = json.loads(source.read_text())
rows = []
for run in r['runs']:
    states = run.get('production_state', [])
    native = [s for rank in run.get('ranks', []) for s in rank.get('native_steps', [])]
    rows.append(dict(name=run['name'], policy=run['policy'], active=run['active'],
        performance_qualified=run.get('performance_qualified', False),
        model_candidate_accepted=run.get('model_candidate_accepted', False),
        quality_pass=run['quality_pass'], bitwise_bridge_native=run['bitwise_token_match'],
        text=run['text'], timing=run.get('serving_timing', run.get('diagnostic_timing')),
        stream_capture_gate=run.get('moe_stream_capture_gate'),
        rank_stream_counters=[s.get('moe_stream', {}).get('python_capture_calls', {}) for s in states],
        native_median_us={k: statistics.median(s[k] for s in native)/1000 for k in
            ('native_wall_ns', 'enqueue_ns', 'synlaunch_ns', 'hccl_ns', 'caller_ns') if native and all(k in s for s in native)},
        native_plan_info=[rank.get('captures', [{}])[0].get('plan_info') for rank in run.get('ranks', []) if rank.get('captures')]))
summary = dict(status=r['status'], candidate_accepted=r['candidate_accepted'],
    model_scope=r.get('model_scope'), model_layers=70 if r.get('model_scope') == 'full_model' else None,
    result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), runs=rows,
    quality={k:v for k,v in r.get('fp32_quality', {}).items() if k != 'audits'},
    teacher_checks=[{k:x[k] for k in ('policy','position','quality_role','check') if k in x}
                    for x in r.get('teacher_forced', {}).get('rows', [])],
    chats=r.get('chat_quality', []),
    batch_comparisons=r.get('batch_policy_abba', {}).get('comparisons', []),
    batch_status=r.get('batch_policy_abba', {}).get('status'),
    batch_runs=[{k:x[k] for k in ('batch','arm','trial','policy','quality_pass','repeat_match','performance_qualified',
                'full_batch_decode_timing','prefill_completion_timing','moe_stream_capture_gate') if k in x}
                for x in r.get('batch_runs', [])],
    error=r.get('error'), scope='Actual vLLM MiMo TP8. Timing remains diagnostic unless the unchanged full quality gates pass.')
if a.executor and r.get('quality_plan') and r.get('chat_quality'):
    sys.path.insert(0, str(a.executor/'executor'))
    from fp32_quality_contract import classify
    summary['independent_quality_audit'] = classify(r, r['quality_plan'], (0, 16, 48), 70)
    summary['independent_quality_scope'] = 'Read-only evaluation of the original thresholds; original run status and acceptance remain unchanged.'
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:summary[k] for k in ('status','candidate_accepted','quality','batch_status','error')},ensure_ascii=False,indent=2))
