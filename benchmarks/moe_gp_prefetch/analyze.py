"""Complete MoE production ABBA and engine traces, with no model extrapolation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as stats
import sys

p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
records=[]
for root in a.roots:
    for case in sorted((root/'results').iterdir()):
        if not (case/'exit.json').exists():continue
        status=json.loads((case/'exit.json').read_text())
        if not any('moe_gp_prefetch/device.py' in v for v in status['command']):continue
        record=dict(root=str(root),case=case.name,exit_code=status['runner_exit_code'],passed=False)
        records.append(record)
        if status['runner_exit_code']!=0:continue
        result=json.loads((case/'result.json').read_text())
        assert result['status']=='PASS_FULL_MOE_BITS_ABBA'
        assert all(r.get('bad',0)==0 and r.get('consumer_bits_equal',True) and r.get('production_prefetch_bitwise',True)for r in result['checks'])
        paths=list((case/'post_graph.json').rglob('*.json')) if (case/'post_graph.json').is_dir() else [case/'post_graph.json']
        mapping={};graphs=[]
        for f in paths:
            for g in json.loads(f.read_text())['graphs']:
                nodes=[n for n in g['nodes']if not n.get('is_logical',False)]
                gp=[n for n in nodes if n['guid']in('gk_moe_gp_scale_tail_v1','gk_moe_gp_prefetch_experiment')]
                if not gp:continue
                assert len(gp)==1 and len(nodes)==6
                tensors={t['name']:t for t in g['tensors']}
                x=tensors[gp[0]['input_tensors'][2]]
                assert x['dtype']=='bf16' and x['max_shape'][0]==6144
                t=x['max_shape'][1];mode='production' if gp[0]['guid']=='gk_moe_gp_scale_tail_v1' else 'prefetch'
                mapping[g['name']]=(t,mode)
                graphs.append(dict(name=g['name'],T=t,mode=mode,physical_nodes=len(nodes),guids=[n['guid']for n in nodes]))
        assert len(mapping)==2*len({r['T']for r in result['checks']})
        record.update(passed=True,checks=len(result['checks']),geometry=result['geometry'],graphs=graphs,
                      libraries=result['libraries'],result_sha256=hashlib.sha256((case/'result.json').read_bytes()).hexdigest())
        profiled=(case/'profile.json').exists()
        record['profiled']=profiled
        if not profiled:
            pairs=defaultdict(list)
            for t,state in sorted({(r['T'],r['state'])for r in result['timing']}):
                trials=[]
                for trial in sorted({r['trial']for r in result['timing']}):
                    rows=[r for r in result['timing']if (r['T'],r['state'],r['trial'])==(t,state,trial)]
                    assert [r['mode']for r in rows]==['production','prefetch','prefetch','production']
                    old=stats.mean(r['event_us']for r in rows if r['mode']=='production')
                    new=stats.mean(r['event_us']for r in rows if r['mode']=='prefetch')
                    trials.append(dict(trial=trial,production_us=old,candidate_us=new,latency_reduction=1-new/old))
                old=stats.mean(r['production_us']for r in trials);new=stats.mean(r['candidate_us']for r in trials)
                pairs[(t,state)]=dict(T=t,state=state,production_us=old,candidate_us=new,latency_reduction=1-new/old,trials=trials)
            record['ABBA']=list(pairs.values())
        else:
            files=list((case/'trace').rglob('*.json'))
            # Keep merged B/E trace only; profiler config and per-invocation
            # analysis JSON are not engine events.
            files=[f for f in files if 'traceEvents' in json.loads(f.read_text())]
            assert files, 'profile produced no engine trace'
            traces=[]
            for f in files:
                parsed=analyze(f,case/'post_graph.json',list(mapping))
                summary=[]
                for name,(t,mode)in mapping.items():
                    rows=[r for r in parsed['invocations']if r['recipe']==name]
                    if not rows:continue
                    ops=defaultdict(list);cores=defaultdict(set)
                    for r in rows:
                        for n in r['nodes']:
                            ops[n['op']].append(n['envelope_us']);cores[n['op']].add(n['engine_count'])
                    summary.append(dict(T=t,mode=mode,invocations=len(rows),envelope_median_us=stats.median(r['envelope_us']for r in rows),
                                        node_envelope_median_us={k:stats.median(v)for k,v in ops.items()},
                                        observed_engine_counts={k:sorted(v)for k,v in cores.items()}))
                traces.append(dict(file=str(f),summary=summary,**parsed))
            record['traces']=traces
result=dict(records=records,scope='Paired full TP-local MoE graph replays and physical-node profiles. Synthetic fixture, original production defaults unchanged; no whole-model TPS claim.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in records:
    print(json.dumps({k:v for k,v in r.items()if k in('case','passed','ABBA')}))
    for t in r.get('traces',[]):print(json.dumps(t['summary']))
