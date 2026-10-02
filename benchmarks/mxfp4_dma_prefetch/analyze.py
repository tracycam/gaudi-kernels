"""Paired real MXFP4 direct-HBM vs packed-DMA-SRAM experiment audit."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sram_ports'))
from analyze import trace


def read_graph(case):
    p=case/'post_graph.json';p=next(p.glob('*.json')) if p.is_dir() else p
    gs=json.loads(p.read_text())['graphs'];assert len(gs)==1
    return gs[0],hashlib.sha256(p.read_bytes()).hexdigest()


def audit(case):
    status=json.loads((case/'exit.json').read_text())
    assert status['module']==5 and status['source_identity_verified']
    if status['runner_exit_code']!=0:return dict(case=case.name,pass_=False,exit=status['runner_exit_code'])
    rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{')]
    p=next(r for r in rows if r.get('stage')=='plan');pf=next(r for r in rows if r.get('stage')=='prefetch')
    layout=next(r for r in rows if r.get('stage')=='scratch_layout')
    assert p['mode']=='mme' and not p['bias'];ms=list(map(int,p['M'].split(',')))
    g,sha=read_graph(case);t={v['name']:v for v in g['tensors']};ns=[n for n in g['nodes'] if not n['is_logical']]
    counts=collections.Counter(n['engine'] for n in ns)
    active=sum(m>0 for m in ms);assert p['decode_nodes']==p['mme_nodes']==active
    assert counts=={'TPC':active,'MME':active,**({'DMA':2*active} if pf['packed_to_sram'] else {})}
    ds=[n for n in ns if n['engine']=='TPC'];copies=[n for n in ns if n['engine']=='DMA']
    producers={n['output_tensors'][0]:n for n in copies};original=[];scratch=set();decoded=set()
    for n in ds:
        assert n['guid']=='gk_mxfp4_decode_bf16_v1' and n['params']==[0,0,0,0]
        w,s,lut=(t[v] for v in n['input_tensors']);y=t[n['output_tensors'][0]]
        assert y['allocation']=='SRAM' and y['dtype']=='bf16' and y['max_shape']==[p['N'],p['K']]
        assert y['rmw_section'] and not y['persistent'];decoded.add(y['name'])
        scratch.add((y['offset'],y['offset']+math.prod(y['max_shape'])*2,'decoded'))
        for v,kind in [(w,'packed'),(s,'scale')]:
            assert v['dtype']=='uint8'
            if pf['packed_to_sram']:
                assert v['allocation']=='SRAM' and not v['persistent'] and v['rmw_section']
                copy=producers[v['name']];assert len(copy['input_tensors'])==1
                source=t[copy['input_tensors'][0]]
                assert source['max_shape']==v['max_shape'] and source['dtype']=='uint8'
                scratch.add((v['offset'],v['offset']+math.prod(v['max_shape']),kind))
            else:source=v
            assert source['allocation']=='DRAM' and source['persistent']
            original.append(source['name'])
    assert len(set(original))==len(original)==active*2
    original_bytes=sum(math.prod(t[n]['max_shape']) for n in original)
    assert original_bytes==p['logical_weight_bytes']
    spans=sorted(scratch)
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:])),spans
    assert sum(b-a for a,b,_ in spans)==p['scratch_bytes']<=16*1024**2
    for n in ns:
        if n['engine']=='MME':
            a,w=(t[v] for v in n['input_tensors']);y=t[n['output_tensors'][0]]
            assert w['name'] in decoded and a['dtype']=='bf16' and y['dtype']=='float32'
        elif n['engine']=='DMA':assert not decoded.intersection(n['input_tensors'])
    result=dict(case=case.name,pass_=True,source_commit=status['git_commit'],plan=p,prefetch=pf,layout=layout,
                graph_sha256=sha,physical_nodes=dict(counts),scratch_intervals=spans,
                original_input_bytes=original_bytes,original_hbm_sources_each_referenced_once=True,
                expanded_weights_only_sram=True,physical_hbm_transactions_counted=False,
                tpc_library_sha256=status['kernel_libraries_sha256'],
                mme_strategies=sorted(set(n.get('mme_node_strategy','') for n in ns if n['engine']=='MME')))
    if any(r.get('stage')=='compile_only' for r in rows):result['compile_only']=True;return result
    gate=next(r for r in rows if r.get('stage')=='correctness');assert gate['bad']==0 and gate['checked']==sum(ms)*p['N']
    samples=[r['event_us'] for r in rows if r.get('stage')=='timing'];assert len(samples)==5
    result.update(correctness=gate,samples_us=samples,event_us=statistics.median(samples),
                  effective_original_TBps=original_bytes/statistics.median(samples)/1e6,
                  tensors_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in case.glob('*.bin')})
    traces=list((case/'trace').glob('*_7.json'));assert len(traces)<=1
    if traces:
        q=trace(traces[0],tpc_main_name=None);nodes=q['nodes'];dec=[n for n in nodes if n['engine']=='TPC' and n['name'].endswith('_decode')]
        q['decode_node_median_us']=statistics.median(n['end_us']-n['start_us'] for n in dec)
        q['steady_decode_node_median_us']=statistics.median(n['end_us']-n['start_us'] for n in dec[2:]) if len(dec)>2 else None
        result['profile']=q
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    manifest=json.loads((args.root/'ASSET-SHA256.json').read_text())
    for n,d in manifest.items():assert hashlib.sha256((args.root/n).read_bytes()).hexdigest()==d,n
    rows=[audit(c) for c in sorted((args.root/'results').iterdir()) if (c/'exit.json').exists()]
    by_name={r['case']:r for r in rows};comparisons=[]
    for label,e,m,n,k in json.loads((args.root/'planned-shapes.json').read_text()):
        arms=[by_name[f'abba-{label}-{i}-p{pf}'] for i,pf in enumerate([0,1,1,0])]
        pairs=[(0,1),(3,2)]
        if f'baab-{label}-0-p1' in by_name:
            arms.extend(by_name[f'baab-{label}-{i}-p{pf}'] for i,pf in enumerate([1,0,0,1]))
            pairs.extend([(5,4),(6,7)])
        assert all(r['pass_'] and 'profile' not in r for r in arms)
        assert all(r['tpc_library_sha256']==arms[0]['tpc_library_sha256'] for r in arms)
        assert all(r['mme_strategies']==arms[0]['mme_strategies'] for r in arms)
        input_names=[s for s in arms[0]['tensors_sha256'] if '_output.bin' not in s]
        assert all(all(r['tensors_sha256'][s]==arms[0]['tensors_sha256'][s] for s in input_names) for r in arms)
        equal=all(r['tensors_sha256']==arms[0]['tensors_sha256'] for r in arms)
        d=statistics.mean(r['event_us'] for r in arms if not r['prefetch']['packed_to_sram']);s=statistics.mean(r['event_us'] for r in arms if r['prefetch']['packed_to_sram'])
        comparisons.append(dict(shape=label,E=e,M=m,N=n,K=k,direct_us=d,dma_prefetch_us=s,latency_change_percent=(s/d-1)*100,
                                direct_original_TBps=arms[0]['original_input_bytes']/d/1e6,prefetch_original_TBps=arms[0]['original_input_bytes']/s/1e6,
                                all_inputs_outputs_oracles_bitwise_equal=equal,tensor_files_compared=len(arms[0]['tensors_sha256']),
                                arm_us=[r['event_us'] for r in arms],pair_latency_change_percent=[(arms[l]['event_us']/arms[a]['event_us']-1)*100 for a,l in pairs],
                                direct_scratch_bytes=arms[0]['plan']['scratch_bytes'],prefetch_scratch_bytes=arms[1]['plan']['scratch_bytes']))
    result=dict(scope='Same ELF and precision, real MXFP4 W4A16 linear operator; not fastest grouped kernel or model TPS',
                asset_files_verified=len(manifest),comparisons=comparisons,records=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in comparisons:print(json.dumps(r))
