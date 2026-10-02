"""Compiler logical byte coverage and SRAM audit; no physical bus claim.

Descriptor parser reused from the preserved original-FP8 placement audit.
"""
import argparse
import itertools
import json
from pathlib import Path
import re

def parse_node(text):
    name=re.search(r'^  name: "([^"]+)"',text,re.M).group(1)
    op=re.search(r'^  op: "([^"]+)"',text,re.M).group(1);attrs={}
    for attr in re.findall(r'\n  attr \{\n.*?\n  \}',text,re.S):
        key=re.search(r'key: "([^"]+)"',attr)
        string=re.search(r'\bs: ("(?:[^"\\]|\\.)*")',attr)
        if key and string:attrs[key.group(1)]=json.loads(string.group(1))
    return name,op,attrs

def tensor_info(s,elem):
    shape=list(map(int,re.search(r'Sizes = \[([^]]+)\]',s).group(1).split(',')))
    strides=list(map(int,re.search(r'strides = \[([^]]+)\]',s).group(1).split(',')))[1:]
    assert len(shape)==len(strides)
    offset=re.search(r'isAliased = ([^,]+), offset: (\d+)',s)
    if offset:assert offset.group(1).strip() in ['prepared_weight'],s
    base=int(offset.group(2)) if offset else 0
    span=elem;i=len(shape)-1
    while i>=0 and strides[i]==span:span*=shape[i];i-=1
    intervals=[]
    for index in itertools.product(*(range(size) for size in shape[:i+1])):
        start=base+sum(v*stride for v,stride in zip(index,strides));intervals.append((start,start+span))
    return shape,intervals

def coverage(intervals,total):
    cursor=0
    for begin,end in sorted(intervals):
        if begin!=cursor:return False
        cursor=end
    return cursor==total

def audit(path,record):
    nodes=[parse_node(t) for t in re.split(r'^node \{\n',path.read_text(),flags=re.M)[1:]]
    descriptors={v.split('  |')[0].strip():v for _,_,attrs in nodes for key,v in attrs.items() if 'Tensor:' in key}
    def origin(value):
        seen=set();offset=0
        while True:
            name=value.split('  |')[0].strip()
            assert name not in seen,name
            seen.add(name)
            alias=re.search(r'isAliased = ([^,]+), offset: (\d+)',value)
            if not alias:return name,offset
            parent=alias.group(1).strip();offset+=int(alias.group(2))
            if parent not in descriptors:return parent,offset
            value=descriptors[parent]
    logical={'Placeholder','Split','Concatenate','Reshape','OutputTensor'}
    external={'prepared_weight','activation','prepared_channel_scales','bias','output'}
    intervals=[];outputs=[];ops={};reads=[]
    for name,op,attrs in nodes:
        ops[op]=ops.get(op,0)+1
        if op in logical:continue
        for key,value in attrs.items():
            if 'Tensor:' not in key:continue
            base,offset=origin(value)
            if key.startswith('inputTensor:') and base=='prepared_weight' and 'location = in DRAM' in value:
                normalized=re.sub(r'isAliased = [^,]+, offset: \d+',f'isAliased = {base}, offset: {offset}',value)
                shape,regions=tensor_info(normalized,1);intervals+=regions
                reads.append({'node':name,'op':op,'shape':shape,'bytes':sum(e-b for b,e in regions)})
            if key.startswith('outputTensor:') and base not in external:
                outputs.append({'node':name,'op':op,'descriptor':value,'sram':'location = in SRAM' in value})
    total=record['N']*record['K']
    result={'case':path.parts[-3],'logical_weight_bytes':sum(e-b for b,e in intervals),
        'expected_weight_bytes':total,'logical_weight_coverage_once':coverage(intervals,total),
        'all_intermediates_in_sram':bool(outputs) and all(v['sram'] for v in outputs),
        'node_counts':ops,'weight_reads':reads,'intermediates':outputs,'physical_bus_bytes':None}
    result['pass']=result['logical_weight_coverage_once'] and result['all_intermediates_in_sram']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    results=[]
    for file in sorted(a.directory.glob('*/result.json')):
        rec=json.loads(file.read_text())
        graphs=list((file.parent/'graphs').glob('*PostGraph-symbol.pbtxt'))
        if rec['returncode']==0:
            assert len(graphs)==1,(file,graphs)
            results.append(audit(graphs[0],rec))
    (a.directory/'placement.json').write_text(json.dumps(results,indent=2))
    for r in results:print(r['case'],r['pass'],r['logical_weight_bytes'],len([v for v in r['intermediates'] if not v['sram']]))
    assert results
