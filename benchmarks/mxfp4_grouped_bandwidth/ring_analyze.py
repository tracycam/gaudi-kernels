"""Local audit of real grouped graph rewrites, including discarded controls."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import numpy as np
from analyze import digest, graph_audit
from isa_identity import text_section
spec=importlib.util.spec_from_file_location('port_trace',Path(__file__).resolve().parents[1]/'sram_ports/analyze.py')
port=importlib.util.module_from_spec(spec);spec.loader.exec_module(port)

def shape(p):return tuple(p[k] for k in ('experts','pool','N','K','M_cap','banks','pattern'))

def placement(case,p):
    path=case/'post_graph.json';path=next(path.glob('*.json')) if path.is_dir() else path
    g=json.loads(path.read_text())['graphs'][0];ts={t['name']:t for t in g['tensors']}
    ns=[n for n in g['nodes'] if not n['is_logical']]
    ds=[n for n in ns if n['guid']=='gk_grouped_mxfp4_decode'];ms=[n for n in ns if n['engine']=='MME']
    gates=[n for n in ns if n['guid']=='gk_grouped_ring_gate']
    assert ds and len(ds)==len(ms) and len(ns)==len(ds)+len(ms)+len(gates)
    for name in ('packed','scales'):assert ts[name]['persistent'] and ts[name]['dtype']=='uint8' and ts[name]['allocation']=='DRAM'
    visited=[];spans=[];buffers=[];names=set()
    for n in ds:
        assert n['input_tensors'][:2]==['packed','scales']
        idx=ts[n['input_tensors'][3]];t=ts[n['output_tensors'][0]];start=(idx['offset']-ts['expert_ids']['offset'])//4;count=idx['max_shape'][0]
        assert idx['allocation']=='DRAM' and idx['alias_of']=='expert_ids'
        assert t['max_shape']==[p['N'],p['K'],count] and t['allocation']=='SRAM' and t['dtype']=='bf16' and not t['persistent']
        visited.extend(range(start,start+count));b=2*math.prod(t['max_shape']);spans.append((t['offset'],t['offset']+b));names.add(t['name'])
        buffers.append(dict(node=n['name'],tensor=t['name'],expert_start=start,expert_count=count,address=t['offset'],bytes=b))
    assert sorted(visited)==list(range(p['experts']))
    for n in ms:assert len(set(n['input_tensors'])&names)==1 and ts[n['output_tensors'][0]]['dtype']=='float32'
    for gate in gates:
        assert len(set(gate['input_tensors'][1:])&names)==3
        assert ts[gate['output_tensors'][0]]['max_shape']==[p['K'],p['M_cap'],p['experts']]
    union=port.merge(spans);capacity=sum(b-a for a,b in union);assert capacity<=48*1024*1024
    return dict(graph_sha256=digest(path),all_expanded_weights_sram=True,expert_coverage_once=True,
                physical_nodes=len(ns),decoder_nodes=len(ds),mme_nodes=len(ms),gate_nodes=len(gates),
                unique_buffer_addresses=len(set(a for a,b in spans)),sram_address_union_bytes=capacity,
                compiled_blocking_edges=sum(len(n.get('blocking_nodes',[])) for n in ns),
                mme_strategies=sorted(set(n.get('mme_node_strategy','') for n in ms)),buffers=buffers)

p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[];baseline={};pending=[];isa=[]
for root in a.roots:
    obj=root/'build/decode.o';isa.append(dict(root=str(root),decode_text_sha256=hashlib.sha256(text_section(obj)).hexdigest()))
    for case in sorted((root/'results').iterdir()):
        status=json.loads((case/'exit.json').read_text());assert status['module']==5 and status['source_identity_verified']
        rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{"stage":')]
        r=dict(root=str(root),case=case.name,returncode=status['runner_exit_code'],source_commit=status['git_commit'])
        if status['runner_exit_code']:
            r['accepted']=False;records.append(r);continue
        plan=next(x for x in rows if x['stage']=='plan');r['plan']=plan
        try:r['placement']=placement(case,plan)
        except AssertionError:
            assert any(x['stage']=='compile_only' for x in rows),'Executed candidate failed placement gate'
            path=case/'post_graph.json';path=next(path.glob('*.json')) if path.is_dir() else path
            g=json.loads(path.read_text())['graphs'][0];ts={t['name']:t for t in g['tensors']}
            decoded=[ts[n['output_tensors'][0]] for n in g['nodes'] if n['guid']=='gk_grouped_mxfp4_decode']
            r.update(accepted=False,compile_only=True,rejection='placement/coverage gate failed; no device replay',
                     rejected_decoded=[dict(name=t['name'],allocation=t['allocation'],bytes=2*math.prod(t['max_shape'])) for t in decoded],graph_sha256=digest(path))
            records.append(r);continue
        meta=next((x for x in rows if x['stage']=='manual_ring'),None)
        if meta:r['requested_ring']=meta
        if any(x['stage']=='compile_only' for x in rows):r['compile_only']=True;records.append(r);continue
        check=next(x for x in rows if x['stage']=='correctness');assert check['bad']==0
        r.update(correctness=check,event_median_us=statistics.median(x['event_us'] for x in rows if x['stage']=='timing'))
        r['output_sha256']={f.name:digest(f) for f in case.glob('output_bank*.bin')}
        count=plan['experts']*plan['N']*plan['K'];us=r['event_median_us']
        r.update(payload_TBps=count*.5/us/1e6,original_TBps=count*17/32/us/1e6,expanded_feed_over_complete_time_TBps=count*2/us/1e6)
        if meta is None:baseline[shape(plan)]=case
        pending.append((case,r))
        traces=list((case/'trace').glob('*_7.json'))
        if traces:
            tr=port.trace(traces[0],None);r['profile']=tr
            dec=sorted((n for n in tr['nodes'] if n['engine']=='TPC' and 'decode' in n['name']),key=lambda n:n['start_us'])
            mme=sorted((n for n in tr['nodes'] if n['engine']=='MME'),key=lambda n:n['start_us'])
            tr['first_three_decodes_finish_us']=max(n['end_us'] for n in dec[:3])
            tr['first_mme_event_start_us']=mme[0]['start_us']
            tr['first_three_finish_before_mme']=tr['first_three_decodes_finish_us']<=tr['first_mme_event_start_us']
            tr['mme_event_start_intervals_us']=[b['start_us']-x['start_us'] for x,b in zip(mme,mme[1:])]
            tr['mme_event_gaps_us']=[max(0,b['start_us']-x['end_us']) for x,b in zip(mme,mme[1:])]
            tr['mme_event_gap_sum_us']=sum(tr['mme_event_gaps_us'])
            if meta and meta.get('data_prefill'):
                gs=[n for n in tr['nodes'] if n['engine']=='TPC' and 'data_prefill' in n['name']]
                assert len(gs)==1 and tr['first_three_finish_before_mme']
                assert gs[0]['start_us']>=tr['first_three_decodes_finish_us']
                tr['data_prefill_confirmed']=True
        records.append(r)
for case,r in pending:
    b=baseline[shape(r['plan'])]
    for name in ('packed.bin','scales.bin','activation.bin','expert_ids.bin'):
        assert digest(case/name)==digest(b/name),(case,name)
    for i in range(r['plan']['banks']):
        f=f'output_bank{i}.bin';assert (case/f).read_bytes()==(b/f).read_bytes(),(case,f)
        y=np.fromfile(case/f,dtype='<f4');ref=np.fromfile(case/f'oracle_bank{i}.bin',dtype='<f8')
        assert y.size==r['plan']['experts']*r['plan']['N']*r['plan']['M_cap'] and np.isfinite(y).all()
        assert np.linalg.norm(y-ref)/max(np.linalg.norm(ref),1e-30)<2e-6
    r.update(accepted=True,bitwise_equal_to_auto=True,baseline_path=str(b))
assert len({x['decode_text_sha256'] for x in isa})==1,'host graph comparison must keep decoder instructions identical'
by={r['case']:r for r in records};abba=[]
for e in (8,16,32):
    names=[f'abba-e{e}-{i}-{n}' for i,n in enumerate(('auto','ring2','ring2','auto'))]
    arms=[by[n] for n in names];assert all(x.get('accepted') and 'profile' not in x for x in arms)
    before=statistics.mean(arms[i]['event_median_us'] for i in (0,3));after=statistics.mean(arms[i]['event_median_us'] for i in (1,2));count=e*512*6144
    abba.append(dict(experts=e,M=2,N=512,K=6144,arms=names,baseline_us=before,candidate_us=after,
                     latency_reduction=1-after/before,payload_TBps=count*.5/after/1e6,
                     original_TBps=count*17/32/after/1e6,expanded_feed_TBps=count*2/after/1e6))
out=dict(scope='Single-device W4A16 projection; original logical weight reads, not bus counters or model TPS',
         decoder_isa=isa,ABBA=abba,records=records,
         caveat='Requested control edges and slot counts are not proofs. Use compiled allocation and trace; MME markers can be writeback milestones, so bandwidth uses full unprofiled event time.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(abba,indent=2))
