"""Derive paired endpoint rates; keep profile-only estimates distinctly labeled."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

p=argparse.ArgumentParser();p.add_argument('measurements',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=json.loads(a.measurements.read_text());rows=data['records']
r5={r['case']:r for r in rows if Path(r['root']).name=='r5'}
fields=('tpc_read_bytes','tpc_write_bytes','mme_read_bytes','mme_write_bytes')
rates=[]
for prefix in ('abba-read','abba-write','abba-copy','abba-mme-bf16','abba-mme-fp8','abba-mme-ab','abba-mixed','balanced-mixed'):
    arms=[r5[f'{prefix}-{i}'] for i in range(4)]
    assert all(r['full_output_oracle_pass'] and 'profile' not in r for r in arms)
    short_us=statistics.mean(arms[i]['event_median_us'] for i in (0,3))
    long_us=statistics.mean(arms[i]['event_median_us'] for i in (1,2));delta=long_us-short_us
    assert delta>0
    volumes={k:arms[1]['plan'][k]-arms[0]['plan'][k] for k in fields}
    assert all(arms[0]['plan'][k]==arms[3]['plan'][k] and arms[1]['plan'][k]==arms[2]['plan'][k] for k in fields)
    bw={k.replace('_bytes',''):v/delta/1e6 for k,v in volumes.items()}
    paired=[]
    for s,l in ((0,1),(3,2)):
        dt=arms[l]['event_median_us']-arms[s]['event_median_us']
        paired.append(sum(volumes.values())/dt/1e6)
    rates.append(dict(experiment=prefix,cases=[r['case'] for r in arms],short_us=short_us,long_us=long_us,delta_us=delta,
                      incremental_bytes=volumes,incremental_TBps=bw,total_incremental_TBps=sum(bw.values()),
                      pair_total_TBps_range=[min(paired),max(paired)],
                      complete_long_TBps={k.replace('_bytes',''):arms[1]['plan'][k]/long_us/1e6 for k in fields},
                      complete_long_total_TBps=sum(arms[1]['plan'][k] for k in fields)/long_us/1e6,
                      denominator='Difference of complete synEvent times; ABBA mean of per-process five-sample medians. Each sample averages 20 replays.'))
fits=[]
for dtype in ('bf16','fp8'):
    rs=[r5[f'cold-{dtype}-k{k}'] for k in (512,2048,4096,8192)]
    x=np.array([r['plan']['mme_read_bytes']/1e6 for r in rs]);y=np.array([r['profile']['mme_span_us'] for r in rs])
    slope,intercept=np.polyfit(x,y,1)
    assert all(r['plan']['mme_nodes']==1 and all(m['marker']=='START_EVENT' for m in r['profile']['markers'] if m['engine']=='MME' and m['phase']=='B') for r in rs)
    fits.append(dict(experiment='cold-single-pass-'+dtype,cases=[r['case'] for r in rs],bytes=[r['plan']['mme_read_bytes'] for r in rs],mme_span_us=y.tolist(),
                     fitted_TBps=1/slope,intercept_us=intercept,max_residual_us=float(max(abs(y-(slope*x+intercept)))),
                     scope='Separate profile, whole single-MME start-to-final-writeback envelope; excludes initialization DMA. Cross-check, not the paired main timing.'))
rs=[r5[f'write-sweep-b{i}'] for i in (1,2,3,4)]
x=np.array([r['plan']['mme_write_bytes']/1e6 for r in rs]);y=np.array([r['profile']['mme_span_us'] for r in rs]);slope,intercept=np.polyfit(x,y,1)
fits.append(dict(experiment='mme-fp32-output',cases=[r['case'] for r in rs],bytes=[r['plan']['mme_write_bytes'] for r in rs],mme_span_us=y.tolist(),
                 fitted_TBps=1/slope,intercept_us=intercept,max_residual_us=float(max(abs(y-(slope*x+intercept)))),
                 scope='Profile-only write-path estimate; full graph additionally drains the entire output to HBM. Not measured concurrently with MME read peak.'))
result=dict(measurements_sha256=hashlib.sha256(a.measurements.read_bytes()).hexdigest(),
            numerically_verified_runs=sum(r.get('full_output_oracle_pass',False) for r in rows),
            source_revisions=sorted(set(r['source_commit'] for r in rows)),paired_rates=rates,profile_cross_checks=fits,
            simultaneous_main_case='balanced-profile',
            limitation='Logical requested bytes with exact output, assembly and placement audits. No complete SRAM-controller transaction counter; these are achieved endpoint rates, not proof of the aggregate physical peak.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in rates:print(r['experiment'],r['incremental_TBps'],r['total_incremental_TBps'],r['pair_total_TBps_range'])
for r in fits:print(r['experiment'],r['fitted_TBps'],r['max_residual_us'])
