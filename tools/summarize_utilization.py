"""Offline, reproducible useful-work throughput ratios from saved complete recipes.

No accelerator access. Ratios use nominal peaks, not measured bus traffic or busy
cycles. Keep every successful candidate, including slower/rejected alternatives.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
sources={};rows=[]


def read(path):
    data=(root/path).read_bytes();sources[path]=hashlib.sha256(data).hexdigest();return json.loads(data)


def add(family,engine,m,n,k,us,wall,weight_bytes,prepared_bytes,source,case,variant,**extra):
    assert us>0 and m>0 and n>0 and k>0
    peak={'MME BF16':432.,'MME FP8':865.,'TPC BF16':11.}.get(engine)
    bw=weight_bytes/us/1e6;tf=2*m*n*k/us/1e6
    rows.append(dict(family=family,engine=engine,M_total=m,N=n,K=k,event_us=us,wall_us=wall,
        useful_weight_bytes_once=weight_bytes,prepared_weight_bytes_once=prepared_bytes,
        useful_weight_TB_s=bw,nominal_HBM_payload_pct=100*bw/2.45,
        useful_TFLOPS=tf,nominal_engine_TFLOPS=peak,
        nominal_compute_pct=None if peak is None else 100*tf/peak,
        source=source,case=case,variant=variant,**extra))


path='evidence/bf16-coverage/summary.json'
for case in read(path)['cases']:
    for r in case['records']:
        if r['returncode'] or not r['correctness']['screen_pass']:continue
        engine={'mme':'MME BF16','tpc':'TPC FP32','tpc-native':'TPC BF16'}[r['backend']]
        m,n,k=r['m'],r['n'],r['k'];t=r['timing']
        add('BF16 W16A16',engine,m,n,k,t['median_event_us'],t['median_wall_us'],2*n*k,r['layout']['weight_bytes'],path,case['case'],r['label'],
            output=r['dtype'],bias=r['bias'],split=r['split'],input_kind=r['kind'],M_groups=str(m),numerical_pass=True)

selected=read('evidence/fp8-coverage/measured-plans.json')['plans']
for file in sorted((root/'evidence/fp8-coverage').glob('*/summary.json')):
    path=str(file.relative_to(root));case=file.parent.name
    for r in read(path):
        if r['returncode'] or not all(v['pass'] for v in r['rows'] if v['stage']=='correctness'):continue
        m,n,k=r['M'],r['N'],r['K']
        accepted=any(v['case']==case and v['shape']==[m,n,k] and v['activation_policy']==r['mode'] and v['input_kind']==r['kind'] for v in selected)
        add('FP8 '+r['mode'].upper(), 'MME FP8' if r['mode']=='w8a8' else 'MME BF16',m,n,k,
            r['event_median_us'],r['wall_median_us'],n*k,n*k,path,case,r['kind'],
            output='bf16',bias=1,input_kind=r['kind'],selected_plan=accepted,M_groups=str(m),
            scale_bytes_excluded=4*n,numerical_pass=True)

path='evidence/mxfp4-coverage/summary.json'
for r in read(path)['records']:
    plan=r.get('plan',{})
    if not r.get('complete_pass') or plan.get('mode') not in ['mme','tpc']:continue
    ms=r['host_known_expert_rows'];n,k=plan['N'],plan['K'];active=sum(v>0 for v in ms)
    # Count checkpoint nibbles + E8M0, including the required last odd-K nibble.
    useful=active*(n*((k+1)//2)+n*((k+31)//32))
    add('MXFP4 W4A16','MME BF16' if plan['mode']=='mme' else 'TPC BF16',sum(ms),n,k,
        r['event_us'],r['synchronized_wall_us'],useful,r['weight_storage_active_bytes'],path,r['case'],
        'Ntile='+str(plan['n_tile']),output='f32',bias=int(plan['bias']),input_kind=plan['pattern'],
        M_groups=','.join(map(str,ms)),active_experts=active,
        logical_weight_read_bytes=r['logical_weight_read_bytes'],numerical_pass=True,
        repeated_GEMV_weight_traversals=sum(ms) if plan['mode']=='tpc' else active)

# Historical real-QKV fixtures: different scale contract and earlier runs.
for mode,m,us,path in [
    ('W8A8',1,20.28865,'evidence/reports/MAC-ROUTES-RESULTS-20260926.md'),
    ('W8A8',16,29.925,'evidence/reports/MAC-ROUTES-RESULTS-20260926.md'),
    ('W8A16',1,32.5464,'evidence/reports/MAC-ROOFLINE-RESULTS-20260926.md'),
    ('W8A16',64,37.53855,'evidence/reports/MAC-ROOFLINE-RESULTS-20260926.md'),
    ('W8A16',513,82.64285,'evidence/reports/MAC-ROOFLINE-RESULTS-20260926.md')]:
    sources[path]=hashlib.sha256((root/path).read_bytes()).hexdigest()
    add('Block FP8 '+mode,'MME FP8' if mode=='W8A8' else 'MME BF16',m,3392,6144,us,None,
        3392*6144,3392*6144,path,'historical-real-qkv','split512+1' if m==513 else 'whole',
        output='bf16',bias=0,input_kind='real QKV fixture',M_groups=str(m),numerical_pass=True,
        original_scale_bytes_excluded=5184)

columns=list(dict.fromkeys(k for r in rows for k in r))
with (out/'all-measurements.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
metadata={'status':'CALCULATED_FROM_SAVED_RUNS','accelerator_used':False,'rows':len(rows),
    'HBM_TB_s':2.45,'MME_BF16_TFLOPS':432,'MME_FP8_TFLOPS':865,'TPC_BF16_vector_TFLOPS':11,
    'compute_ratio':'useful 2*M*N*K divided by complete operator time and relevant nominal engine peak; not busy cycles',
    'bandwidth_ratio':'unique useful weight payload / complete operator time / nominal HBM peak; not physical traffic utilization',
    'bytes':'BF16=2NK; FP8=NK (scales listed separately, excluded); MXFP4=checkpoint packed nibbles+E8M0 for active experts',
    'TPC_FP32_peak':'not assigned; do not divide TPC-only code by MME peak',
    'TPC_BF16_note':'11 TFLOPS is the published BF16 vector reference; MAC_ACC32 throughput was not separately calibrated',
    'official_sources':['https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html','https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf'],
    'source_sha256':sources}
(out/'all-measurements.json').write_text(json.dumps({'metadata':metadata,'records':rows},indent=2)+'\n')
print(json.dumps({k:v for k,v in metadata.items() if k!='source_sha256'},indent=2))
