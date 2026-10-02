"""Inspect compiled graph aliases, weight accesses and intermediate placement.

Descriptor coverage does not measure HBM transactions or TPC inner-loop traffic.
"""
import argparse
import itertools
import json
from pathlib import Path
import re

def nodes(path):
 result=[]
 for text in re.split(r'^node \{\n',path.read_text(),flags=re.M)[1:]:
  name=re.search(r'^  name: "([^"]+)"',text,re.M).group(1);op=re.search(r'^  op: "([^"]+)"',text,re.M).group(1);attrs={}
  for attr in re.findall(r'\n  attr \{\n.*?\n  \}',text,re.S):
   key=re.search(r'key: "([^"]+)"',attr);value=re.search(r'\bs: ("(?:[^"\\]|\\.)*")',attr)
   if key and value:attrs[key.group(1)]=json.loads(value.group(1))
  result.append((name,op,attrs))
 return result

def intervals(value,offset,element=2):
 shape=list(map(int,re.search(r'Sizes = \[([^]]+)\]',value).group(1).split(',')))
 strides=list(map(int,re.search(r'strides = \[([^]]+)\]',value).group(1).split(',')))[1:];assert len(shape)==len(strides)
 span=element;i=len(shape)-1
 while i>=0 and strides[i]==span:span*=shape[i];i-=1
 return [(offset+sum(v*s for v,s in zip(index,strides)),offset+sum(v*s for v,s in zip(index,strides))+span) for index in itertools.product(*(range(d) for d in shape[:i+1]))]

def audit(case):
 records=[]
 for path in sorted(case.glob('*/graphs/*PostGraph-symbol.pbtxt')):
  parsed=nodes(path);descriptors={v.split('  |')[0].strip():v for _,_,a in parsed for key,v in a.items() if 'Tensor:' in key}
  def origin(value):
   seen=set();offset=0
   while True:
    name=value.split('  |')[0].strip();assert name not in seen;seen.add(name)
    alias=re.search(r'isAliased = ([^,]+), offset: (\d+)',value)
    if not alias:return name,offset
    parent=alias.group(1).strip();offset+=int(alias.group(2))
    if parent not in descriptors:return parent,offset
    value=descriptors[parent]
  reads=[];intermediates=[];outputs=[];counts={};ranges=[]
  for name,op,attrs in parsed:
   counts[op]=counts.get(op,0)+1
   if op in {'Placeholder','Split','Concatenate','Reshape','OutputTensor'}:continue
   for key,value in attrs.items():
    if 'Tensor:' not in key:continue
    base,offset=origin(value)
    if key.startswith('inputTensor:') and base.startswith('weight_bf16') and 'location = in DRAM' in value:
     regions=intervals(value,offset);ranges.extend(regions);reads.append({'node':name,'op':op,'base':base,'offset':offset,'bytes':sum(e-b for b,e in regions),'descriptor':value})
    if key.startswith('outputTensor:'):
     record={'node':name,'op':op,'base':base,'offset':offset,'descriptor':value,'sram':'location = in SRAM' in value}
     if base=='output':outputs.append(record)
     elif base not in {'activation_bf16','bias_fp32','weight_bf16_nk','weight_bf16_kn'}:intermediates.append(record)
  ordered=sorted(ranges);union=0;end=0
  for b,e in ordered:
   union+=max(0,e-max(end,b));end=max(end,e)
  logical=sum(e-b for b,e in ranges)
  records.append({'case':path.parts[-3],'node_counts':counts,'weight_descriptor_bytes':logical,'weight_union_bytes':union,'descriptor_read_multiple':logical/union if union else None,'weight_reads':reads,'intermediates':intermediates,'dram_intermediate_count':sum(not v['sram'] for v in intermediates),'output_writes_alias_external_output':bool(outputs),'output_writes':outputs,'physical_HBM_bytes':None})
 return records
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.case)
 a.output.write_text(json.dumps({'records':r,'scope':'compiled graph descriptor/alias audit, not hardware bus transactions'},indent=2)+'\n')
 for x in r:print(x['case'],x['node_counts'],'weight descriptor multiplier',x['descriptor_read_multiple'],'DRAM intermediates',x['dram_intermediate_count'])
 assert r
