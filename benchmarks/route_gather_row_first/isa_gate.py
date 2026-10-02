"""Execute the real old/new glue+ELF geometry on all output bytes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for n in ('old-library','new-library','runner','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);records=[]
def run(name,x,mapping,bad,r,c,slot,batch):
 expected=np.zeros((batch,c,6144),np.uint16)
 if not bad:
  for e in range(batch):
   for row in range(c):
    q=mapping[(slot+e)*c+row]
    if 0<=q<len(x)*r:expected[e,row]=x[q//r]
 expected.tofile(out/(name+'-cpu.bin'));outputs={}
 for mode,lib in (('old',a.old_library),('row_first',a.new_library)):
  folder=out/(name+'-'+mode);folder.mkdir();spec=['gk_route_tile_gather_bf16_v1','3','1','3',str(r),str(c),str(slot)]
  for i,array in enumerate((x,mapping,np.array([bad],np.int32))):
   f=folder/f'input{i}.bin';array.tofile(f);spec.extend(['bf16'if i==0 else'i32',str(array.ndim),*map(str,array.shape[::-1]),str(f)])
  actual=folder/'actual.bin';spec.extend(['bf16','3','6144',str(c),str(batch),str(actual)]);(folder/'spec.txt').write_text(' '.join(spec)+'\n')
  cmd=[str(a.runner.resolve()),str(lib.resolve()),str(folder/'spec.txt'),'-'];(folder/'command.json').write_text(json.dumps(cmd)+'\n')
  with(folder/'run.log').open('wb')as log:subprocess.run(cmd,env={**os.environ,'TPC_RUNNER':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
  outputs[mode]=np.fromfile(actual,np.uint16).reshape(expected.shape)
 row=dict(case=name,T=len(x),R=r,C=c,B=batch,slot=slot,status=bad,words=expected.size,old_cpu_bad=int((outputs['old']!=expected).sum()),new_cpu_bad=int((outputs['row_first']!=expected).sum()),pair_bad=int((outputs['old']!=outputs['row_first']).sum()))
 records.append(row);(out/'partial.json').write_text(json.dumps(records,indent=2)+'\n');assert not any(row[k]for k in ('old_cpu_bad','new_cpu_bad','pair_bad')),row
for t,c,batch,slot,r in [(1,1,1,0,1),*[(17,3,3,2,r)for r in range(1,9)],(513,32,2,0,8),(130,16,5,1,3)]:
 x=((np.arange(t*6144,dtype=np.uint32)*79+31)%65536).astype(np.uint16).reshape(t,6144)
 mapping=((np.arange((slot+batch+1)*c)*31+7)%(t*r)).astype(np.int32)
 if len(mapping)>3:mapping[slot*c]=-1;mapping[slot*c+1]=t*r
 run(f't{t}-c{c}-b{batch}-r{r}',x,mapping,0,r,c,slot,batch)
run('bad-status',x,mapping,9,r,c,slot,batch)
report=dict(status='PASS_REAL_GLUE_AND_ISA_ROW_FIRST',records=records,word_pairs=sum(r['words']for r in records),libraries={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in (a.old_library,a.new_library,a.runner)},device_tested=False)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('records','libraries')}))
