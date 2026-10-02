"""Audit static-group public graphs: complete fixtures, 1x prepared reads, no expanded HBM."""
import argparse
import itertools
import json
import math
from pathlib import Path


def audit(path):
    expected=json.loads((path.parent/'result.json').read_text())['records']
    assert expected and all(r['numerical_pass'] for r in expected)
    records=[]
    for graph in json.loads(path.read_text())['graphs']:
        tensors={t['name']:t for t in graph['tensors']};experts={}
        def origin(t):
            seen=set()
            while t.get('alias'):
                assert t['name'] not in seen;seen.add(t['name']);t=tensors[t['alias_of']]
            return t
        def intervals(t):
            base=origin(t);span=1;i=0
            assert t['dtype_bit_size']==8 and t['allocation']=='DRAM'
            while i<len(t['max_shape']) and t['strides'][i]==span:span*=t['max_shape'][i];i+=1
            return [(t['offset']-base['offset']+sum(v*s for v,s in zip(index,t['strides'][i:])),span)
                    for index in itertools.product(*(range(n) for n in t['max_shape'][i:]))]
        def trace(name,seen):
            if name in seen:return []
            seen.add(name);t=tensors[name]
            assert t['allocation']=='SRAM' and not t['persistent'] and t['dtype']=='bf16',('expanded HBM weight',name)
            readers=[]
            for alias in tensors.values():
                if alias.get('alias_of')==name:readers+=trace(alias['name'],seen)
            for node in graph['nodes']:
                if name not in node['input_tensors']:continue
                if node['guid']=='gemm':
                    assert node['input_tensors'][1]==name
                    readers.append(origin(tensors[node['input_tensors'][0]])['max_shape'][1])
                else:
                    for output in node['output_tensors']:readers+=trace(output,seen)
            return readers
        for node in graph['nodes']:
            if node['guid']!='gk_mxfp4_decode_bf16_v1':continue
            assert node['params']==[0,0,0,0],'public full-N decoder offset changed'
            w,s=(tensors[name] for name in node['input_tensors'][:2]);wr,sr=origin(w),origin(s)
            e=experts.setdefault(wr['name'],{'N':0,'K':wr['max_shape'][1],'readers':[],'regions':{},'expected':{}})
            for t,base in [(w,wr),(s,sr)]:
                assert base['persistent'] and base['allocation']=='DRAM'
                e['regions'].setdefault(base['name'],[]).extend(intervals(t));e['expected'][base['name']]=math.prod(base['max_shape'])
            for name in node['output_tensors']:
                e['N']+=tensors[name]['max_shape'][0];e['readers']+=trace(name,set())
        if not experts:continue
        for e in experts.values():
            assert e['readers'] and len(set(e['readers']))==1
            e['M']=e['readers'][0]
            for name,regions in e['regions'].items():
                cursor=0
                for offset,size in sorted(regions):assert offset==cursor,('weight reread/hole',name);cursor+=size
                assert cursor==e['expected'][name]
            e['prepared_read_bytes']=sum(e['expected'].values());e.pop('regions')
        shapes={(e['N'],e['K']) for e in experts.values()};assert len(shapes)==1
        n,k=next(iter(shapes))
        records.append({'graph':graph['name'],'N':n,'K':k,'active_M':sorted(e['M'] for e in experts.values()),
                        'experts':list(experts.values()),'decoded_sram_only':True,'prepared_weight_and_scales_coverage_once':True,
                        'workspace_size':graph['workspace_size'],'physical_hbm_transactions':'not_measured'})
    actual=sorted((r['N'],r['K'],r['active_M']) for r in records)
    wanted=sorted((r['N'],r['K'],sorted(m for m in r['M_groups'] if m)) for r in expected)
    assert actual==wanted,('incomplete grouped fixture coverage',actual,wanted)
    return {'status':'PASS','records':records}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('post_graph',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.post_graph);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
