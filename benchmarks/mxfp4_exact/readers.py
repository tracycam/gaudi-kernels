"""Check compact reader values and logical byte intervals against the loader."""
import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--loader',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('compact_exact_verification',a.loader)
loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
lib=ctypes.CDLL(str(a.library.resolve()));lib.reader_v3.argtypes=[ctypes.c_int]+[ctypes.c_uint32]*4;lib.reader_v3.restype=ctypes.c_uint32
lib.reader_pair.argtypes=[ctypes.c_int,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_int];lib.reader_pair.restype=ctypes.c_uint32
rng=np.random.default_rng(2709);cases=[]
for n,k in [(1,1),(3,2),(127,31),(128,32),(129,33),(511,255),(512,256),(513,257),(1023,31),(1024,32),(1025,33)]:
 rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8)
 scales=rng.integers(0,255,(n,(k+31)//32),dtype=np.uint8)
 prepared=loader.prepare(rows,scales,logical_k=k)
 weight_reads,scale_reads=Counter(),Counter();terms=0;outputs=[]
 for pair in range(n//512*256+n%512):
  columns=[lib.reader_pair(i,pair,n,1) for i in range(3)]
  columns=columns[:columns[2]];outputs.extend(columns);previous=[None,None]
  for z in range(k):
   shared=None
   for lane,col in enumerate(columns):
    offset,shift,scale=[lib.reader_v3(i,col,z,n,k) for i in range(3)]
    if offset!=previous[lane] and offset!=shared:weight_reads[offset]+=1
    previous[lane]=offset;shared=offset
    if z%32==0:scale_reads[scale]+=1
    actual=(int(prepared.weight[offset])>>shift)&15
    expected=(int(rows[col,z//2])>>(4*(z%2)))&15
    assert actual==expected and prepared.scales[scale]==scales[col,z//32]
    terms+=1
 assert sorted(outputs)==list(range(n)) and terms==n*k
 assert set(weight_reads)==set(range(prepared.weight.size)) and set(weight_reads.values())=={1}
 assert set(scale_reads)==set(range(prepared.scales.size)) and set(scale_reads.values())=={1}
 cases.append({'N':n,'K':k,'products_checked':terms,'packed_bytes':len(weight_reads),'scale_bytes':len(scale_reads),'logical_read_max_per_byte':1})
result={'status':'CPU_reader_and_logical_intervals_only_no_physical_HBM_measurement','all_pass':True,'loader_path':str(a.loader.resolve()),'loader_sha256':hashlib.sha256(a.loader.read_bytes()).hexdigest(),'cases':cases}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(cases),'all_pass':True,'products':sum(c['products_checked'] for c in cases)}))
