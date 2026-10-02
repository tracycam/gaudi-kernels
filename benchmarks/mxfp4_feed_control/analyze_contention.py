"""Strict graph/whole-output checks for the synthetic SRAM contention controls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import struct
from analyze import profile


def audit(case):
    rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{')]
    plan=next(r for r in rows if r.get('stage')=='plan')
    dependency=next((r for r in rows if r.get('stage')=='dependency'),{})
    outcome=json.loads((case/'exit.json').read_text())
    assert outcome['returncode']==outcome['runner_exit_code']==0
    assert outcome['source_identity_verified'] and outcome['module']==5
    assert '--hard-control' in outcome['command']
    p=case/'post_graph.json'
    if p.is_dir():
        files=list(p.glob('*.post.json'));assert len(files)==1;p=files[0]
    graphs=json.loads(p.read_text())['graphs'];assert len(graphs)==1
    g=graphs[0];t={v['name']:v for v in g['tensors']}
    physical=[n for n in g['nodes'] if not n['is_logical']]
    assert all(n['engine'] in ('TPC','MME') for n in physical)
    producers=[n for n in physical if n['engine']=='TPC'];consumers=[n for n in physical if n['engine']=='MME']
    E,N,K,M,c,slots=(plan[k] for k in ('E','N','K','M','chunk','slots'))
    resident=plan['schedule']=='resident';dtype='bf16' if plan['mode']=='bf16' else 'hfloat8'
    expected_gen=slots if resident else E//c
    assert len(producers)==expected_gen and len(consumers)==E//c
    outputs=[t[n['output_tensors'][0]] for n in producers]
    weights=[t[n['input_tensors'][1]] for n in consumers]
    assert all(w['allocation']=='SRAM' and not w['persistent'] and w['dtype']==dtype
               and w['max_shape']==[N,K,c] for w in outputs+weights)
    offsets=sorted(set(w['offset'] for w in outputs))
    assert len(offsets)==slots
    if not dependency.get('separate_sections'):
        assert all(offsets[i]-offsets[0]==i*(plan['tile_bytes']+plan['padding_bytes']) for i in range(slots))
    assert all(offsets[i+1]-offsets[i]>=plan['tile_bytes'] for i in range(slots-1))
    assert plan['generated_weight_bytes']==expected_gen*plan['tile_bytes']
    assert plan['consumed_weight_bytes']==len(consumers)*plan['tile_bytes']
    assert len({n['input_tensors'][0] for n in consumers})==len(consumers)
    assert all(n['guid'] in ('gemm','batch_gemm') and n['params']==[0,0] for n in consumers)
    assert all(t[n['output_tensors'][0]]['dtype']=='float32' for n in consumers)
    result=dict(case=case.name,plan=plan,pass_=True,source_commit=outcome['git_commit'],
                dependency=dependency,
                all_expanded_weights_sram=True,no_dma=True,generator_count=len(producers),mme_count=len(consumers),
                offsets=offsets,graph_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                mme_strategies=sorted(set(n.get('mme_node_strategy','') for n in consumers)))
    if any(r.get('stage')=='compile_only' for r in rows):return result
    gate=next(r for r in rows if r.get('stage')=='correctness')
    assert gate['bad']==0 and gate['checked']==E*N*M
    values=(0,.5,1,1.5,2,3,4,6)
    for i in range(E//c):
        actual=struct.unpack('<'+'f'*(c*M*N),(case/f'output_{i}_checked.bin').read_bytes())
        source=i%slots if resident else i
        for e in range(c):
            for m in range(M):
                for n in range(N):
                    ref=values[(n+(source*c+e)*3)%8]*(-1 if (n//8+source*c+e+i)&1 else 1)*K*(m+1)
                    assert actual[(e*M+m)*N+n]==ref
    times=[r for r in rows if r.get('stage')=='timing'];assert len(times)==5
    result.update(correctness=gate,independent_binary_oracle_pass=True,samples=times,
                  event_median_us=statistics.median(r['event_us'] for r in times),
                  wall_median_us=statistics.median(r['wall_us'] for r in times))
    traces=list((case/'trace').glob('*_7.json'))
    if traces:
        assert len(traces)==1
        prof=profile(traces[0]);result['profile']=prof
        ts=[n for n in prof['node_envelopes'] if n['engine']=='TPC'];ms=[n for n in prof['node_envelopes'] if n['engine']=='MME']
        result['mme_node_median_us']=statistics.median(n['end_us']-n['start_us'] for n in ms)
        result['logical_consumer_TBps']=plan['consumed_weight_bytes']/prof['mme_union_us']/1e6
        if resident:
            assert max(n['end_us'] for n in ts)<=min(n['start_us'] for n in ms)
            result['no_tpc_execution_during_mme']=True
        if plan['schedule']=='serial':
            result['serial_verified']=prof['spu_mme_overlap_us']<0.01
            if dependency.get('serial_data_fence'):
                assert result['serial_verified'], 'data-fenced serial control still overlaps'
            else:
                result['serial_control_qualified']=result['serial_verified']
        # Bus samples are retained raw: marker names/units alone do not prove
        # transferred byte counts or provide a bank-conflict counter.
        es=json.loads(traces[0].read_text())['traceEvents']
        procs={e['pid']:e['args']['name'] for e in es if e['ph']=='M' and e['name']=='process_name'}
        counts=collections.Counter(e.get('name') for e in es if e.get('ph')=='C'
                                   and procs.get(e.get('pid'),'').startswith('Bus monitors'))
        result['bus_counter_sample_counts']=dict(counts)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.root/'ASSET-SHA256.json').read_text())
    for n,d in manifest.items():assert hashlib.sha256((a.root/n).read_bytes()).hexdigest()==d,n
    rs=[]
    for r in sorted((a.root/'results').iterdir()):
        if not (r/'exit.json').exists():continue
        status=json.loads((r/'exit.json').read_text())
        if status['runner_exit_code']!=0:
            rs.append(dict(case=r.name,pass_=False,runner_exit_code=status['runner_exit_code'],reason='payload failed; retain raw logs, exclude from performance'))
        else:rs.append(audit(r))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(scope='Synthetic SRAM control, not MXFP4 throughput',asset_hashes_verified=len(manifest),records=rs),indent=2)+'\n')
    print(json.dumps([dict(case=r['case'],event_us=r.get('event_median_us'),mme_node_us=r.get('mme_node_median_us'),consumer_TBps=r.get('logical_consumer_TBps')) for r in rs],indent=2))
