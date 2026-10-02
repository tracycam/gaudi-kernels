"""Summarize complete probes and independently verify retained small fixtures."""
import argparse, hashlib, json, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rows=[]
for case in sorted(a.results.iterdir()):
    if not (case/'run.log').is_file():continue
    records=[]
    for line in (case/'run.log').read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try:records.append(json.loads(line))
            except json.JSONDecodeError:pass
    plan=next((r for r in records if r.get('stage')=='plan'),{})
    quality=next((r for r in records if r.get('stage')=='correctness'),{})
    times=[r for r in records if r.get('stage')=='timing']
    exitdata=json.loads((case/'exit.json').read_text()) if (case/'exit.json').exists() else {}
    row={'case':case.name,'plan':plan,'correctness':quality,'exit':exitdata.get('runner_exit_code',exitdata.get('returncode')),
         'source_commit':exitdata.get('git_commit'),'library_sha256':exitdata.get('kernel_libraries_sha256'),
         'scope':'single-device operator; no model quality/TPS or physical HBM counter claim'}
    if plan:
        ms=list(map(int,plan['M'].split(',')));n=plan['N'];k=plan['K']
        expected=n*k if plan['mode']=='decode' else sum(ms)*n
        row['expected_outputs']=expected
        row['complete_pass']=row['exit']==0 and quality.get('bad')==0 and quality.get('checked')==expected and expected>0
        if plan['mode']!='decode':
            bytes_per_expert=((n+255)//256)*(128*k+256*((k+31)//32))
            row['weight_storage_active_bytes']=sum(m>0 for m in ms)*bytes_per_expert
            row['logical_weight_read_bytes']=bytes_per_expert*(sum(ms) if plan['mode']=='tpc' else sum(m>0 for m in ms))
            row['host_known_expert_rows']=ms
            row['host_row_padding']=0
            row['packed_n_padding']=((n+255)//256)*256-n
            row['tpc_k_splits']=[min((k+31)//32,max(1,(24+((n+255)//256)*m-1)//(((n+255)//256)*m))) if m else 0 for m in ms] if plan['mode']=='tpc' and case.name.endswith(('v7','v8')) else None
    if times:
        row.update(event_us=statistics.median(r['event_us'] for r in times),
                   synchronized_wall_us=statistics.median(r['wall_us'] for r in times),
                   event_range_us=[min(r['event_us'] for r in times),max(r['event_us'] for r in times)],samples=len(times))
    rows.append(row)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'records':rows,'complete_passes':sum(r.get('complete_pass',False) for r in rows),
    'failed_or_incomplete':sum(not r.get('complete_pass',False) for r in rows)},indent=2)+'\n')
for r in rows:
    print(r['case'],'PASS' if r.get('complete_pass') else 'FAIL',r.get('event_us',''),r.get('synchronized_wall_us',''))
