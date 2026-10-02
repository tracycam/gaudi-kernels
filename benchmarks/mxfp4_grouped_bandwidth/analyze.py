"""Audit grouped projections, rejected experiments, and physical engine traces.

Bandwidth is original packed+scale bytes divided by complete event latency.
It is not a bus-counter measurement, full MoE timing, or model TPS claim.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mxfp4_pipeline'))
from isa_identity import text_section
from trace_overlap import merge


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20), b''):h.update(b)
    return h.hexdigest()


def graph_audit(case, plan):
    path=case/'post_graph.json'
    if path.is_dir():
        paths=list(path.glob('*.post.json'));assert len(paths)==1;path=paths[0]
    graphs=json.loads(path.read_text())['graphs'];assert len(graphs)==1
    graph=graphs[0];ts={t['name']:t for t in graph['tensors']}
    e,pool,n,k,m,s,banks=(plan[a] for a in ('experts','pool','N','K','M_cap','splits','banks'))
    tile=plan.get('weight_tile',2*ts['packed']['max_shape'][0])
    assert 8<=e<=pool<=384 and pool==e*banks and n%tile==0 and k%32==0
    for name,shape in [('packed',[tile//2,k,n//tile,pool]),('scales',[tile,k//32,n//tile,pool])]:
        t=ts[name];assert t['persistent'] and t['dtype']=='uint8' and t['allocation']=='DRAM'
        assert t['max_shape']==shape
        assert (case/(name+'.bin')).stat().st_size==math.prod(shape)
    assert plan['rotating_weight_bytes']==pool*n*k*17//32
    assert plan['weight_bytes_per_replay']==e*n*k*17//32
    raw=(case/'expert_ids.bin').read_bytes();stride=max(256,((e*4+255)//256)*256)
    ids=[struct.unpack_from('<'+str(e)+'i',raw,b*stride) for b in range(banks)]
    assert all(len(set(x))==e and all(0<=a<pool for a in x) for x in ids)
    assert sorted(a for x in ids for a in x)==list(range(pool))
    nonlogical=[x for x in graph['nodes'] if not x.get('is_logical',False)]
    result=dict(pass_=True,graph_sha256=digest(path),physical_nodes=len(nonlogical),
                guids=[x['guid'] for x in nonlogical],unique_experts_each_replay=True,
                rotation_disjoint=True,original_bytes_per_replay=plan['weight_bytes_per_replay'],
                rotating_bytes=plan['rotating_weight_bytes'],
                scope='Graph placement and source-loop element requests; physical HBM transactions not measured')
    if plan['mode']=='tpc':
        assert sum(x['guid']=='gk_grouped_mxfp4_smallm' for x in nonlogical)==1
        assert all(x['guid'] in ('gk_grouped_mxfp4_smallm','gk_grouped_mxfp4_reduce') for x in nonlogical)
        assert all(t['dtype']!='bf16' or t['name'].startswith(('activation','byte_lut')) for t in ts.values())
        assert len(nonlogical)==1+(s>1)
        chunk=(k//32+s-1)//s
        assert [g for z in range(s) for g in range(z*chunk,min((z+1)*chunk,k//32))]==list(range(k//32))
        # Each weight loop is outside the M row MACs. Literal prefetch reads
        # 2 + 64*30 + 63*2 = 2048 K entries/task, never a speculative final tile.
        result.update(no_expanded_weight_tensor=True,split_k_partition=True)
    else:
        dec=[x for x in nonlogical if x['guid']=='gk_grouped_mxfp4_decode']
        mme=[x for x in nonlogical if x['engine']=='MME']
        assert dec and len(mme)==len(dec) and len(nonlogical)==len(dec)+len(mme)
        visited=[];decoded=set();spans=[]
        for node in dec:
            assert node['input_tensors'][:2]==['packed','scales']
            idx=ts[node['input_tensors'][3]];out=ts[node['output_tensors'][0]]
            assert idx.get('alias_of','expert_ids')=='expert_ids'
            start=(idx['offset']-ts['expert_ids']['offset'])//4;count=idx['max_shape'][0]
            visited.extend(range(start,start+count))
            assert out['max_shape']==[n,k,count] and out['dtype']=='bf16' and out['allocation']=='SRAM'
            assert not out['persistent'];decoded.add(out['name'])
            spans.append((out['offset'],out['offset']+2*n*k*count))
        assert sorted(visited)==list(range(e))
        for node in mme:
            assert len(set(node['input_tensors']) & decoded)==1
            assert ts[node['output_tensors'][0]]['dtype']=='float32'
        assert all(t['allocation']=='SRAM' for t in ts.values() if t['name'].startswith('decoded_weights'))
        result.update(all_expanded_weights_sram=True,decoder_nodes=len(dec),mme_nodes=len(mme),
                      decoded_sram_address_union_bytes=sum(b-a for a,b in merge(spans)))
    return result


def trace(path):
    events=json.loads(path.read_text())['traceEvents']
    names={e['pid']:e['args']['name'] for e in events if e['ph']=='M' and e['name']=='process_name'}
    pending=collections.defaultdict(list);spans=collections.defaultdict(list)
    for e in events:
        engine=next((s for s in ('TPC','MME') if names.get(e.get('pid'),'').startswith('*'+s)),None)
        if not engine or e['ph'] not in ('B','E'):continue
        a=e.get('args',{});op=a.get('op','')
        if not(op.startswith('gk_grouped_mxfp4_') or op.lower() in ('gemm','batch_gemm')):continue
        key=(e['pid'],e['tid'],e['name'],a.get('id'),a.get('Unique Node ID'),a.get('recipe id'))
        if e['ph']=='B':pending[key].append(e['ts'])
        else:
            assert pending[key];spans[(engine,op)].append((pending[key].pop(0),e['ts']))
    assert spans and not any(pending.values())
    all_spans=merge([x for v in spans.values() for x in v]);origin=all_spans[0][0]
    ops=[]
    for (engine,op),v in spans.items():
        union=merge(v)
        ops.append(dict(engine=engine,op=op,event_pairs=len(v),start_us=union[0][0]-origin,
                        end_us=union[-1][1]-origin,union_us=sum(b-a for a,b in union)))
    return dict(path=str(path),sha256=digest(path),ops=ops,
                span_us=all_spans[-1][1]-origin,idle_us=all_spans[-1][1]-origin-sum(b-a for a,b in all_spans))


def analyze(root):
    cases=[];isa=[];traces=[];seals=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():continue
        seal=folder/'seal-summary.json'
        if not seal.exists():continue # Do not analyze an incomplete transfer.
        seals.append(json.loads(seal.read_text()))
        manifest=json.loads((folder/'remote-sha256.json').read_text())
        for log in sorted(folder.glob('results/*/run.log')):
            rows=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{"stage":')]
            if not rows:continue
            plan=next(r for r in rows if r['stage']=='plan')
            check=next((r for r in rows if r['stage']=='correctness'),None)
            times=[r['event_us'] for r in rows if r['stage']=='timing']
            out=json.loads((log.parent/'exit.json').read_text())
            profile=(log.parent/'profile.json').exists()
            hashes={p.name:manifest[str(p.relative_to(folder))] for p in log.parent.glob('*.bin')}
            record=dict(stage=folder.name,case=log.parent.name,path=str(log.parent),plan=plan,correctness=check,
                        returncode=out['returncode'],samples_us=times,profiled=profile,files_sha256=hashes,
                        compile_only='--compile-only' in out['command'])
            if check and check['bad']==0:
                assert check['checked']==plan['experts']*plan['N']*plan['M_cap']*plan['banks']
            if record['compile_only']:
                record['graph_audit']={'pass_':False,'status':'not_device_qualified',
                                       'reason':'Compile-only case has no saved device input/output fixture'}
            else:
                try:record['graph_audit']=graph_audit(log.parent,plan)
                except (AssertionError,KeyError,ValueError,OSError) as error:
                    record['graph_audit']={'pass_':False,'reason':repr(error)}
            if times and not profile and out['returncode']==0 and check and check['bad']==0:
                us=statistics.mean(times)
                record.update(event_mean_us=us,effective_weight_TBs=plan['weight_bytes_per_replay']/us/1e6,
                              effective_TFs=2*plan['valid_rows']*plan['N']*plan['K']/us/1e6)
            cases.append(record)
        for obj in sorted(folder.glob('builds/*/grouped.o')):
            dis=obj.with_suffix('.o.dis')
            if not dis.exists():continue
            s=dis.read_text();mem=[line.strip() for line in s.splitlines() if re.search(r'\b(?:ld|st)_l(?:_v)?\b',line) and 'mmio' not in line]
            regs=[int(v)+(kind=='D') for kind,v in re.findall(r'\b(V|D)(\d+)\b',s)]
            record=dict(path=str(obj),text_sha256=hashlib.sha256(text_section(obj)).hexdigest(),
                        text_bytes=len(text_section(obj)),max_vector_register=max(regs,default=None),
                        static_local_memory_instructions=len(mem),local_memory_examples=mem[:8])
            sched=obj.parent/'schedule.json'
            if sched.exists():
                record['schedule']=json.loads(sched.read_text())
                asm=obj.with_suffix('.s').read_text()
                body=asm.split('.LBB0_3:\n')[1].split('.LBB0_6:')[0]
                packets=[[v.strip() for v in row.split(';')] for row in body.splitlines()]
                assert all(len(row)==4 for row in packets)
                record['k32_packets']=len(packets)
                record['slot_opcodes']={slot:dict(collections.Counter(row[i].split()[0] for row in packets))
                                       for i,slot in enumerate(('LOAD','SPU','VPU','STORE'))}
            isa.append(record)
        for path in sorted(folder.glob('results/profile*/trace/*_7.json')):traces.append(trace(path))
    abba=[]
    for e in (8,384):
        arms=[next((c for c in cases if c['stage']=='final' and c['case']==f'e{e}-abba-{s}'),None) for s in 'abcd']
        if not all(arms):continue
        assert all(c['returncode']==0 and c['correctness']['bad']==0 for c in arms)
        files=arms[0]['files_sha256']
        assert all(c['files_sha256']==files for c in arms),'ABBA input/reference/output bytes changed'
        a,b,c,d=arms;old=statistics.mean(a['samples_us']+d['samples_us']);new=statistics.mean(b['samples_us']+c['samples_us'])
        abba.append(dict(experts=e,control_us=old,candidate_us=new,latency_reduction_percent=100*(old-new)/old,
                         effective_weight_TBs=a['plan']['weight_bytes_per_replay']/new/1e6,
                         all_inputs_and_outputs_bitwise_equal=True,replays_per_variant=200))
    return dict(cases=cases,isa=isa,traces=traces,abba=abba,seals=seals,
                scope='Synthetic complete grouped linear projection; not full MoE or model acceptance; no physical HBM byte counter')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(result['cases']),abba=result['abba'],traces=result['traces'],
                         graph_audit_failures=[(c['stage'],c['case'],c['graph_audit']) for c in result['cases'] if not c['graph_audit']['pass_']]),indent=2))
