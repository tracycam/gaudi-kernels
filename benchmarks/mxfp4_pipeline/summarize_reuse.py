"""Summarize sealed pipeline/ISA ablations without discarding slow samples."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_managed import audit
from trace_overlap import measure

p=argparse.ArgumentParser()
p.add_argument('archive',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
records=[]; traces=[]; groups=collections.defaultdict(list); seals=[]
for stage in sorted(a.archive.iterdir()):
    seal=stage/'seal-summary.json'
    if not seal.is_file():continue
    seals.append(json.loads(seal.read_text()))
    hashes=json.loads((stage/'remote-sha256.json').read_text())
    for case in sorted((stage/'results').glob('*')):
        if not (case/'exit.json').is_file():continue
        rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{')]
        plan=next((r for r in rows if r.get('stage')=='plan'),None)
        check=next((r for r in rows if r.get('stage')=='correctness'),None)
        outcome=json.loads((case/'exit.json').read_text())
        passed=outcome['returncode']==0 and check and check['checked']>0 and check['bad']==0
        row=dict(stage=stage.name,case=case.name,source_commit=outcome['git_commit'],
                 numerical_pass=bool(passed),plan=plan,correctness=check,
                 compile_only=any(r.get('stage')=='compile_only'for r in rows),
                 profiled='--profile' in outcome['command'],returncode=outcome['returncode'])
        if passed:
            times=[r for r in rows if r.get('stage')=='timing']
            assert len(times)==5
            mean=sum(r['event_us']*r['repeats'] for r in times)/sum(r['repeats'] for r in times)
            row.update(samples=times,event_mean_us=mean,
                       event_median_us=statistics.median(r['event_us']for r in times),
                       event_min_us=min(r['event_us']for r in times),event_max_us=max(r['event_us']for r in times),
                       wall_mean_us=sum(r['wall_us']*r['repeats'] for r in times)/sum(r['repeats'] for r in times))
            if plan['mode']=='mme':
                row['useful_flops']=2*sum(map(int,plan['M'].split(',')))*plan['N']*plan['K']
                row['useful_TFLOPS_mean']=row['useful_flops']/mean/1e6
                row['effective_weight_TBps_mean']=plan['logical_weight_bytes']/mean/1e6
                try:row['placement']=audit(case)
                except (AssertionError,ValueError,KeyError) as e:
                    row['placement']=dict(pass_=False,reason=repr(e),scope='Unsupported/failed placement proof, not a numerical failure')
            key=tuple(str(plan[k])for k in ('mode','M','N','K','pattern','bias'))
            bins={path.name:hashes[str(path.relative_to(stage))]for path in case.glob('*.bin')}
            groups[key].append(dict(stage=stage.name,case=case.name,files=bins))
        records.append(row)
        for path in case.glob('trace/*_7.json'):
            traces.append(dict(stage=stage.name,case=case.name,**measure(path)))
comparisons=[]
for key,arms in groups.items():
    if len(arms)<2:continue
    original=arms[0]['files']
    for arm in arms[1:]:
        shared=set(original)&set(arm['files'])
        assert shared and shared==set(original)==set(arm['files'])
        diff=[name for name in sorted(shared)if original[name]!=arm['files'][name]]
        comparisons.append(dict(geometry=key,baseline=[arms[0]['stage'],arms[0]['case']],candidate=[arm['stage'],arm['case']],
                                checked_files=len(shared),all_input_output_reference_bytes_equal=not diff,different_files=diff))
abba=[]
for stage in ('explicit','final'):
    arms=sorted((r for r in records if r['stage']==stage and r['case'].startswith('stream32-m128-abba-')),
                key=lambda r:r['case'])
    if not arms:continue
    assert len(arms)==4 and all(r['numerical_pass'] and not r['profiled']for r in arms)
    first=(arms[0]['event_mean_us']+arms[3]['event_mean_us'])/2
    second=(arms[1]['event_mean_us']+arms[2]['event_mean_us'])/2
    abba.append(dict(stage=stage,order=[r['case']for r in arms],all_sample_baseline_us=first,
                     all_sample_candidate_us=second,time_ratio=second/first,speedup=first/second,
                     samples_per_arm=5,replays_per_arm=100,no_samples_removed=True))
result=dict(scope='Static host-known per-expert GEMV/GEMM only; synthetic original-width weights; no model/default promotion',
            records=records,trace_overlap=traces,byte_comparisons=comparisons,abba=abba,seals=seals,
            completed_device_cases=len(records),numerical_pass_cases=sum(r['numerical_pass']for r in records),
            compile_only_cases=sum(r['compile_only']for r in records),
            checked_outputs_passing=sum(r['correctness']['checked']for r in records if r['numerical_pass']),
            physical_hbm_byte_counter=False,model_qualified=False)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k]for k in ('completed_device_cases','numerical_pass_cases','checked_outputs_passing','abba')}))
