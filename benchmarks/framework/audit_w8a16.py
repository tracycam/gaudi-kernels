"""Audit actual compiler placements, independently from numerical/runtime gates."""
import argparse
import itertools
import json
import math
from pathlib import Path


def audit(path,expected_path=None):
    expected=json.loads((expected_path or path.parent/'result.json').read_text())
    assert expected and all(r['numerical_pass'] for r in expected),'missing numerical fixture qualification'
    expected_shapes=sorted((r['M'],r['N'],r['K']) for r in expected)
    records=[]
    for graph in json.loads(path.read_text())['graphs']:
        tensors={t['name']:t for t in graph['tensors']}
        decoders=[n for n in graph['nodes'] if n['guid']=='fp8_linear_decode']
        if not decoders:continue
        roots=set();regions=[];outputs=[];descendants=[];mme_readers=[];seen=set()
        def origin(t):
            chain=set()
            while t.get('alias'):
                assert t['name'] not in chain,'alias cycle'
                chain.add(t['name']);t=tensors[t['alias_of']]
            return t
        def trace_weight(name):
            if name in seen:return
            seen.add(name);t=tensors[name]
            descendants.append({'name':name,'allocation':t['allocation'],'persistent':t['persistent'],'dtype':t['dtype']})
            for alias in tensors.values():
                if alias.get('alias_of')==name:trace_weight(alias['name'])
            for node in graph['nodes']:
                if name not in node['input_tensors']:continue
                if node['guid']=='gemm':
                    assert node['input_tensors'][1]==name,'decoded weight used as activation'
                    activation=origin(tensors[node['input_tensors'][0]])
                    mme_readers.append({'name':node['name'],'activation_shape':activation['max_shape']})
                else:
                    for output in node['output_tensors']:trace_weight(output)
        for node in decoders:
            weight=tensors[node['input_tensors'][0]];root=weight
            while root.get('alias'):root=tensors[root['alias_of']]
            roots.add(root['name'])
            assert weight['dtype_bit_size']==8 and weight['allocation']=='DRAM'
            shape=weight['max_shape'];strides=weight['strides'];span=1;i=0
            while i<len(shape) and strides[i]==span:span*=shape[i];i+=1
            for indices in itertools.product(*(range(n) for n in shape[i:])):
                offset=weight['offset']-root['offset']+sum(v*s for v,s in zip(indices,strides[i:]))
                regions.append((offset,offset+span))
            for name in node['output_tensors']:
                tensor=tensors[name]
                outputs.append({'name':name,'allocation':tensor['allocation'],'persistent':tensor['persistent'],
                                'logical_bytes':math.prod(tensor['max_shape'])*tensor['dtype_bit_size']//8})
                trace_weight(name)
        assert len(roots)==1
        root=tensors[next(iter(roots))];expected=math.prod(root['max_shape']);cursor=0;once=True
        row_counts={r['activation_shape'][1] for r in mme_readers}
        assert len(row_counts)==1,'missing or ambiguous MME reader coverage'
        for begin,end in sorted(regions):
            if begin!=cursor:once=False
            cursor=end
        once=once and cursor==expected
        rec={'graph':graph['name'],'weight_shape_fastest_first':root['max_shape'],'weight_bytes':expected,
             'weight_descriptor_read_bytes':sum(b-a for a,b in regions),'weight_coverage_once':once,
             'M':next(iter(row_counts)),
             'decoded_sram_only':bool(descendants) and all(t['allocation']=='SRAM' and not t['persistent'] and t['dtype']=='bf16' for t in descendants),
             'weight_descendants':descendants,'mme_readers':mme_readers,
             'decode_tiles':len(decoders),'decoded_tensors':outputs,'workspace_size':graph['workspace_size'],
             'physical_hbm_transactions':'not_measured'}
        rec['pass']=rec['weight_coverage_once'] and rec['decoded_sram_only'];records.append(rec)
    assert records,'no decoder evidence'
    actual_shapes=sorted((r['M'],r['weight_shape_fastest_first'][1],r['weight_shape_fastest_first'][0]) for r in records)
    assert actual_shapes==expected_shapes,('incomplete/extra compiled fixture coverage',actual_shapes,expected_shapes)
    return {'status':'PASS' if all(r['pass'] for r in records) else 'FAIL','records':records}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('post_graph',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.post_graph);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}));raise SystemExit(result['status']!='PASS')
