"""Full-output latency, executable ISA, and separately collected engine traces.

Physical engine intervals do not establish VPU issue counts or HBM byte counts.
Static issue packets are not measured cycles. No model TPS inference is made.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mxfp4_pipeline'))
from isa_identity import text_section
from trace_overlap import merge
from audit_managed import audit as audit_mme


def graph_audit(case,plan,controls):
    if plan['mode']=='mme':return audit_mme(case)
    path=case/'post_graph.json'
    if path.is_dir():
        paths=list(path.glob('*.post.json'));assert len(paths)==1;path=paths[0]
    graphs=json.loads(path.read_text())['graphs'];assert len(graphs)==1
    graph=graphs[0];tensors={t['name']:t for t in graph['tensors']}
    ms=list(map(int,plan['M'].split(',')));rows=int(controls['GK_SMALLM_ROWS'])
    assert all(m==rows for m in ms) and rows in (1,2,4)
    assert plan['N']%256==0 and plan['K']%32==0,'padding requires a separate qualification'
    nodes=[n for n in graph['nodes']if n['guid']=='gk_mxfp4_gemv_f32_v1']
    assert len(nodes)==len(ms)
    assert all(n['guid']in ('gk_mxfp4_gemv_f32_v1','gk_mxfp4_reduce_f32_v1','reshape')for n in graph['nodes'])
    assert not any(t['dtype']=='bf16'and not t['persistent']for t in tensors.values()),'expanded weight intermediate'
    seen=[];requests=0
    for node in nodes:
        w,s,lut,x=[tensors[name]for name in node['input_tensors']]
        y=tensors[node['output_tensors'][0]]
        assert w['dtype']==s['dtype']=='uint8'and w['persistent']and s['persistent']
        assert x['dtype']=='bf16'and y['dtype']=='float32'
        n,k=plan['N'],plan['K'];assert w['max_shape']==[128,k,n//256]and s['max_shape']==[256,k//32,n//256]
        splits=y['max_shape'][2]if len(y['max_shape'])==3 else 1
        chunk=(k//32+splits-1)//splits
        visited=[g for i in range(splits)for g in range(i*chunk,min((i+1)*chunk,k//32))]
        assert visited==list(range(k//32))
        requests+=n*k//2+n*k//32;seen.extend([w['name'],s['name']])
    assert len(seen)==len(set(seen))and requests==plan['logical_weight_bytes']
    return dict(pass_=True,one_packed_owner_per_node=True,aligned_no_padding=True,
                logical_weight_read_bytes=requests,no_expanded_weight_tensor=True,
                node_guids=[n['guid']for n in graph['nodes']],graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Actual graph owners plus kernel split-K loop partition proof; not physical HBM byte counters')


def trace(path):
    events=json.loads(path.read_text())['traceEvents']
    names={e['pid']:e['args']['name']for e in events if e['ph']=='M'and e['name']=='process_name'}
    pending=collections.defaultdict(list);spans=collections.defaultdict(list)
    for e in events:
        engine=next((s for s in ('TPC','MME')if names.get(e.get('pid'),'').startswith('*'+s)),None)
        if not engine or e['ph'] not in ('B','E'):continue
        a=e.get('args',{});op=a.get('op')
        if not op or not(op.startswith('gk_mxfp4_')or op.lower()=='gemm'):continue
        key=(e['pid'],e['tid'],e['name'],a.get('id'),a.get('Unique Node ID'),a.get('recipe id'))
        if e['ph']=='B':pending[key].append(e['ts'])
        else:
            assert pending[key],key
            spans[(engine,op)].append((pending[key].pop(0),e['ts']))
    assert spans and not any(pending.values())
    all_spans=merge([x for v in spans.values()for x in v]);origin=all_spans[0][0]
    ops=[]
    for (engine,op),v in spans.items():
        union=merge(v)
        ops.append(dict(engine=engine,op=op,event_pairs=len(v),start_us=union[0][0]-origin,
                        end_us=union[-1][1]-origin,union_us=sum(b-a for a,b in union)))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ops=ops,
                span_us=all_spans[-1][1]-origin,idle_us=all_spans[-1][1]-origin-sum(b-a for a,b in all_spans))


def analyze(root):
    cases=[];isa=[];traces=[]
    for log in sorted(root.glob('*/results/*/run.log')):
        rows=[json.loads(s)for s in log.read_text().splitlines()if s.startswith('{"stage":')]
        if not rows:continue
        bystage={r['stage']:r for r in rows if r['stage']!='timing'}
        times=[r['event_us']for r in rows if r['stage']=='timing']
        launch=json.loads((log.parent/'launch.json').read_text())
        record=dict(path=str(log.parent),case=log.parent.name,**bystage,
                    controls=launch['controls'],exit=json.loads((log.parent/'exit.json').read_text())['returncode'],
                    samples_us=times,profiled=(log.parent/'profile.json').exists())
        record['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in log.parent.glob('*.bin')if 'output'not in p.name and 'oracle'not in p.name}
        record['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in log.parent.glob('*output.bin')}
        record['reference_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in log.parent.glob('*oracle_f64.bin')}
        try:record['graph_audit']=graph_audit(log.parent,bystage['plan'],launch['controls'])
        except (AssertionError,KeyError,ValueError,StopIteration,OSError)as error:
            record['graph_audit']={'pass_':False,'reason':repr(error)}
        if times and not record['profiled']:
            us=statistics.mean(times);plan=bystage['plan'];flops=sum(map(int,plan['M'].split(',')))*2*plan['N']*plan['K']
            record.update(event_mean_us=us,event_median_us=statistics.median(times),
                          effective_weight_TBs=plan['logical_weight_bytes']/us/1e6,effective_TFs=flops/us/1e6)
        cases.append(record)
    for obj in sorted(root.glob('*/builds/*/mxfp4_gemv.o')):
        asm=obj.with_suffix('.s');s=asm.read_text().split('\t.type\ttpc_compiler')[0]
        ops=collections.Counter(line.strip().split()[0]for line in s.splitlines()if line.strip()and not line.strip().startswith(('//','.','{','}','$')))
        regs=[int(x)+(kind=='D')for kind,x in re.findall(r'%(V|D)(\d+)',s)]
        isa.append(dict(path=str(obj),text_sha256=hashlib.sha256(text_section(obj)).hexdigest(),
                        text_bytes=len(text_section(obj)),max_vector_register=max(regs),instructions=dict(ops),
                        non_mmio_local=[x.strip()for x in s.splitlines()if re.search(r'\b(?:ld|st)_l(?:_v)?\b',x)and'mmio'not in x]))
    for path in sorted(root.glob('*/results/profile*/trace/*_7.json')):traces.append(trace(path))
    return dict(cases=cases,isa=isa,traces=traces,scope='Synthetic complete linear projection, not full MoE/model; profiler timings excluded')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(result['cases']),'isa':len(result['isa']),'traces':result['traces']},indent=2))
