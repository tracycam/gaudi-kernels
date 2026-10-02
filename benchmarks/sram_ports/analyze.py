"""Audit local endpoint assets. Never equate event-union time to SRAM bus time."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def merge(xs):
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


def trace(path,tpc_main_name='port_traffic'):
    es=json.loads(path.read_text())['traceEvents']
    procs={e['pid']:e['args']['name'] for e in es if e['ph']=='M' and e['name']=='process_name'}
    threads={(e['pid'],e['tid']):e['args']['name'] for e in es if e['ph']=='M' and e['name']=='thread_name'}
    lane_events=collections.defaultdict(collections.Counter)
    starts=collections.defaultdict(list);spans=collections.defaultdict(list);markers=collections.Counter()
    bus=collections.defaultdict(list);spmu=collections.defaultdict(list)
    spmu_sums=collections.defaultdict(collections.Counter)
    for e in es:
        proc=procs.get(e.get('pid'),'')
        if e.get('ph')=='C':
            target=bus if 'Bus monitors' in proc else spmu if 'SPMU' in proc else None
            if target is not None:target[e['name']].append(e)
            if 'SPMU' in proc:
                for k,v in e.get('args',{}).items():
                    if isinstance(v,(int,float)):spmu_sums[proc][k]+=v
        eng=next((k for k in ('TPC','MME','EDMA') if proc.startswith('*'+k)),None)
        if not eng or e['ph'] not in ('B','E'):continue
        key=(e['pid'],e['tid'],e['name'],e.get('args',{}).get('id'))
        markers[(eng,e['ph'],e.get('args',{}).get('HW event name'))]+=1
        if e['ph']=='B':
            starts[key].append(e['ts'])
            if (eng=='TPC' and (tpc_main_name is None or e['name']==tpc_main_name)) or eng=='MME':
                lane_events[eng][threads[(e['pid'],e['tid'])]]+=1
        else:
            assert starts[key],key
            spans[(eng,e['name'])].append([starts[key].pop(0),e['ts']])
    assert all(not q for q in starts.values())
    origin=min(a for ss in spans.values() for a,b in ss)
    nodes=[dict(engine=e,name=n,start_us=min(a for a,b in ss)-origin,end_us=max(b for a,b in ss)-origin) for (e,n),ss in spans.items()]
    tpc=merge([p for (e,n),ss in spans.items() if e=='TPC' and (tpc_main_name is None or n==tpc_main_name) for p in ss])
    mme=merge([p for (e,n),ss in spans.items() if e=='MME' for p in ss])
    dur=lambda ss:sum(b-a for a,b in ss)
    all_spans=[p for ss in spans.values() for p in ss]
    # Useful overlap evidence, not a byte-normalization denominator.
    overlap=dur(tpc)+dur(mme)-dur(merge(tpc+mme))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                engines_span_us=max(b for a,b in all_spans)-origin,
                tpc_spu_span_us=tpc[-1][1]-tpc[0][0] if tpc else 0,
                mme_span_us=mme[-1][1]-mme[0][0] if mme else 0,
                event_union_overlap_us=overlap,nodes=nodes,
                markers=[dict(engine=k[0],phase=k[1],marker=k[2],count=v) for k,v in markers.items()],
                bmon_samples={k:len(v) for k,v in bus.items()},
                bmon_avg_max={k:max(float(e['args'].get('Avg',0)) for e in v) for k,v in bus.items()},
                spmu_samples={k:len(v) for k,v in spmu.items()},
                spmu_sums_by_unit=dict(spmu_sums),
                main_lane_event_counts=dict(lane_events),
                warning='Markers may be writeback milestones, not read-start events. No union-derived bandwidth is reported.')


def audit(case):
    status=json.loads((case/'exit.json').read_text());assert status['module']==5 and status['source_identity_verified']
    if status['runner_exit_code']!=0:return dict(case=case.name,root=str(case.parent.parent),pass_=False,status=status['runner_exit_code'])
    logs=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{')]
    p=next(r for r in logs if r.get('stage')=='plan')
    gpath=case/'post_graph.json';gpath=next(gpath.glob('*.json')) if gpath.is_dir() else gpath
    gs=json.loads(gpath.read_text())['graphs'];assert len(gs)==1;g=gs[0]
    tensors={t['name']:t for t in g['tensors']};physical=[n for n in g['nodes'] if not n['is_logical']]
    ms=[n for n in physical if n['engine']=='MME'];ts=[n for n in physical if n['name']=='port_traffic']
    has_tpc=p['mode']!='mme';has_mme=p['mode']=='mme' or p['mode'].startswith('mix-')
    assert len(ms)==(p['mme_nodes'] if has_mme else 0) and len(ts)==int(has_tpc)
    assert all(n['engine'] in ('TPC','MME','DMA') for n in physical)
    elem=1 if p['dtype']=='fp8' else 2
    if has_tpc:
        op=p['mode'].removeprefix('mix-');t=ts[0]
        if op!='write':assert tensors[t['input_tensors'][0]]['allocation']=='SRAM'
        if op!='read':assert tensors[t['output_tensors'][0]]['allocation']=='SRAM'
    if has_mme:
        for n in ms:
            a,b=(tensors[s] for s in n['input_tensors'][:2]);y=tensors[n['output_tensors'][0]]
            assert a['allocation']==('SRAM' if p['placement'] in ('a','ab') else 'DRAM')
            assert b['allocation']==('SRAM' if p['placement'] in ('b','ab') else 'DRAM')
            assert y['allocation']==('SRAM' if p['placement']=='out' else 'DRAM')
    result=dict(case=case.name,root=str(case.parent.parent),source_commit=status['git_commit'],pass_=True,plan=p,
                graph_sha256=hashlib.sha256(gpath.read_bytes()).hexdigest(),
                mme_strategies=sorted(set(n.get('mme_node_strategy','') for n in ms)),
                physical_nodes=collections.Counter(n['engine'] for n in physical),
                physical_bus_bytes_measured=False)
    if any(r.get('stage')=='compile_only' for r in logs):result['compile_only']=True;return result
    gate=next(r for r in logs if r.get('stage')=='correctness');assert gate['bad']==0
    if has_tpc:
        r,rep,tasks=p['rows'],p['repeats'],p['tasks'];x=np.arange(64*r*tasks,dtype=np.uint32)%251+1
        il='_il' in ts[0]['guid']
        if op=='read':
            ref=x.reshape((r,tasks,64) if il else (tasks,r,64)).sum(axis=0 if il else 1,dtype=np.uint64)*rep
        elif op=='write':
            index=np.arange(x.size,dtype=np.uint32);ref=rep+(index//64%tasks if il else index//(64*r))
        else:ref=x+rep-1
        actual=np.fromfile(case/'tpc_result_checked.bin',dtype='<u4');assert np.array_equal(actual,ref.astype(np.uint32).reshape(-1))
    if has_mme:
        for i in range(p['mme_nodes']):
            actual=np.fromfile(case/f'mme_result_{i}_checked.bin',dtype='<f4')
            assert actual.size==p['N']*p['M']*p['batch']
            ref=p['K']*(1+np.arange(actual.size,dtype=np.uint32)%p['N']%4)
            assert np.array_equal(actual,ref)
    times=[r['event_us'] for r in logs if r.get('stage')=='timing'];assert len(times)==5
    us=statistics.median(times)
    volumes={k:p[k] for k in ('tpc_read_bytes','tpc_write_bytes','mme_read_bytes','mme_write_bytes')}
    result.update(full_output_oracle_pass=True,correctness=gate,samples_us=times,event_median_us=us,
                  complete_recipe_TBps={k.replace('_bytes',''):b/us/1e6 for k,b in volumes.items()},
                  complete_recipe_total_TBps=sum(volumes.values())/us/1e6,
                  denominator='Complete synEvent replay time, including initialization and final drain; bytes exclude initialization/drain.')
    traces=list((case/'trace').glob('*_7.json'));assert len(traces)<=1
    if traces:result['profile']=trace(traces[0])
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('roots',nargs='+',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    rows=[]
    for root in args.roots:
        for case in sorted((root/'results').iterdir()):
            if (case/'exit.json').exists():rows.append(audit(case))
    result=dict(scope='SRAM endpoint probes; achieved throughput is not proof of peak SRAM bandwidth',records=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['case'],r.get('event_median_us'),r.get('complete_recipe_total_TBps'),r.get('profile',{}).get('event_union_overlap_us'))
