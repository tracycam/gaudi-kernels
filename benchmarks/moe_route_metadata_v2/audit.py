"""Static ISA/metadata request account; never estimates device cycles."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from elftools.elf.elffile import ELFFile
from reference import reference

p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--v1-prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(x):return hashlib.sha256(x).hexdigest()
def text(p):return ELFFile(io.BytesIO(p.read_bytes())).get_section_by_name('.text').data()
kernels={}
for name in ('count','prefix','scatter'):
 source=(a.build/(name+'.s')).read_text();spills=[l for l in source.splitlines()if re.search(r'\b(?:ld_l|st_l)_v\b',l)];assert not spills
 if name=='count':assert re.search(r'ld_tnsr\s+%V\d+, 0x1,',source)and not re.search(r'ld_g\s+%V',source)
 kernels[name]=dict(elf_sha256=sha((a.build/(name+'.o')).read_bytes()),text_sha256=sha(text(a.build/(name+'.o'))),no_vlm_spill=True)
assert text(a.build/'prefix.o')==text(a.v1_prefix)
cost=[];t,r,e,c,b=513,8,384,16,616;length=t*r;chunks=(length+63)//64
for mode in ('uniform','hot','skew'):
 ids=[[(i*r+j)%e for j in range(r)]for i in range(t)]
 if mode!='uniform':
  for i in range(t if mode=='hot'else 384):ids[i]=list(range(376,384))
 plan=reference(ids,e,c,b)
 scalar_reads=sum(min(64,length-q*64)for row in plan['chunk_counts']for q,n in enumerate(row)if n)
 cost.append(dict(state=mode,T=t,R=r,E=e,C=c,B=b,chunk_count_shape=[e,chunks],chunk_count_bytes=e*chunks*4,
   v1_count_id_scalar_loads=e*length,v2_count_id_vector_loads=e*(length//64),v2_count_tail_id_scalar_loads=e*(length%64),
   common_row_validation_id_scalar_loads=t*r*(r+1)//2,
   v1_scatter_id_scalar_loads=e*length,v2_scatter_id_scalar_loads=scalar_reads,v2_scatter_chunk_scalar_loads=e*chunks))
report=dict(status='PASS_STATIC_V2_AUDIT',kernels=kernels,prefix_text_matches_v1=True,costs=cost,
 scope='Instruction presence and logical requested loads; no device throughput, time or physical memory-transaction claim')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(cost,indent=2))
