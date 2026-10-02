"""Small immutable index for a completed, sealed full-model experiment."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
case=a.case;name=case.name;x=json.loads((case/'result.json').read_text())
companion=case.parent/(name+'-runner');runner=json.loads((companion/(name+'.exit.json')).read_text())
assert runner.get('returncode') is not None and runner.get('cleanup',{}).get('reaped') and not runner.get('remaining_owned_processes')
arms={}
for r in x.get('runs',[]):
    marker='-native-policy-abba-'
    if marker not in r['name']:continue
    arm=int(r['name'].rsplit(marker,1)[1]);assert arm not in arms
    timing=r.get('serving_timing',r.get('diagnostic_serving_timing',r.get('diagnostic_timing')))
    assert timing and timing['steady_steps']>0 and timing['tps']>0
    arms[arm]=dict(name=r['name'],policy=r['policy'],timing=timing,
                  execution_gate_pass=r.get('execution_gate_pass'),performance_qualified=r.get('performance_qualified'))
assert sorted(arms)==[0,1,2,3]
pooled={}
for role,indices in [('control',(0,3)),('candidate',(1,2))]:
    n=sum(arms[i]['timing']['steady_steps']for i in indices)
    seconds=sum(arms[i]['timing']['steady_steps']/arms[i]['timing']['tps']for i in indices)
    pooled[role]=dict(tps=n/seconds,ms_per_token=seconds*1000/n,steps=n,arms=list(indices),policies=sorted({arms[i]['policy']for i in indices}),performance_qualified=bool(x.get('candidate_accepted') and all(arms[i]['performance_qualified'] for i in indices)))
long=[{k:r.get(k)for k in ('name','policy','serving_timing','diagnostic_serving_timing','diagnostic_timing','execution_gate_pass','performance_qualified','bitwise_token_match')}for r in x.get('long_context',{}).get('runs',[])]
result=dict(case=name,status=x['status'],accepted=x.get('candidate_accepted'),
    quality=x.get('fp32_quality'),chat_quality=x.get('chat_quality'),native_abba=pooled,native_arms=arms,
    native_gain_ms=pooled['control']['ms_per_token']-pooled['candidate']['ms_per_token'],
    remaining_to_100tps_ms=pooled['candidate']['ms_per_token']-10,
    batch_abba={k:x.get('batch_policy_abba',{}).get(k)for k in ('status','control','candidate','order','expect_cross_policy_bitwise','backend','comparisons','scope','model_candidate_accepted')},long_context=dict(context=x.get('long_context',{}).get('context'),status=x.get('long_context',{}).get('status'),runs=long),
    runner={k:runner.get(k)for k in ('returncode','elapsed_s','cleanup','remaining_owned_processes')},
    result_sha256=hashlib.sha256((case/'result.json').read_bytes()).hexdigest(),
    scope='Within-run native ABBA isolates declared policy difference. All loaded libraries are shared across arms. Cross-run differences do not isolate a library update. B>1 bridge aggregate TPS is not B1 native/per-request TPS.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:result[k]for k in ('case','status','accepted','native_abba','native_gain_ms','remaining_to_100tps_ms')},ensure_ascii=False))
