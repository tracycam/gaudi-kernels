"""Resident SWA ABBA accounting: output identity, numerical gates, delivered TPS."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.case/'result.json').read_text());assert r['status']=='COMPLETED_DIAGNOSTIC'
rows=[]
for batch in sorted({x['B']for x in r['runs']}):
    runs=[x for x in r['runs']if x['B']==batch];timed=[x for x in runs if x['phase']=='timed']
    assert [x['variant']for x in timed]==['swa-scalar','swa-quad','swa-quad','swa-scalar']
    assert len(timed)==4 and len(runs)==8
    hashes=[tuple(hashlib.sha256(json.dumps(o['ids'],separators=(',',':')).encode()).hexdigest()for o in x['outputs'])for x in runs]
    same=len(set(hashes))==1;assert same,'generation differs; no paired speedup claim'
    gates=[x['same_prefix_gate']for x in runs if x['phase']=='warm']
    assert len(gates)==4 and all(g['passed']for g in gates)
    samples={v:[]for v in ('swa-scalar','swa-quad')}
    for x in timed:
        m=x['mature'];tps=m['per_request_tps'];assert len(tps)==batch
        samples[x['variant']].append(dict(block=x['block'],per_stream_tps=[tps[k]for k in sorted(tps,key=int)],
                                         aggregate_tps=m['aggregate_tps'],wall_s=m['wall_s'],
                                         interval_ms=list(m['delivery_intervals'].values())[0]['p50_ms']))
    median={v:dict(per_stream_tps=[statistics.median(x['per_stream_tps'][i]for x in seq)for i in range(batch)],
                   aggregate_tps=statistics.median(x['aggregate_tps']for x in seq),
                   interval_ms=statistics.median(x['interval_ms']for x in seq))for v,seq in samples.items()}
    rows.append(dict(B=batch,samples=samples,median=median,all_tokens_identical=same,max_first_request_query_kl=max(g['max_kl']for g in gates),
                     aggregate_speedup=median['swa-quad']['aggregate_tps']/median['swa-scalar']['aggregate_tps']))
result=dict(status='PASS_RESIDENT_ABBA_IDENTITY_AND_GATES',case=str(a.case),source_result_sha256=hashlib.sha256((a.case/'result.json').read_bytes()).hexdigest(),
            mode=r['mode'],drafter=r['drafter'],rows=rows,candidate_accepted=False,
            scope='Same resident 70-layer model only if recorded config says 70. Common mature delivery window, two A/two B samples; first request/query numerical audit. Short output ceiling does not pass complete-answer quality or long-context gates.',
            layers=r['config'].get('hf_overrides',{}).get('num_hidden_layers',70))
a.output.write_text(json.dumps(result,indent=2)+'\n')
for x in rows:print(json.dumps({k:x[k]for k in ('B','median','aggregate_speedup','max_first_request_query_kl')}))
