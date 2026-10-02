"""Graph-replay metadata oracle, including invalid IDs and count overflow."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args()
build=json.loads((a.build/'build.json').read_text())
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0')
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
torch.ops.load_library(str(a.build.resolve()/'gaudi_smallm.so'))
out=Path(os.environ['PROBE_OUT']);records=[]
def sync():hc.mark_step();torch.hpu.synchronize()
def oracle(ids,cap):
    flat=ids.reshape(-1);l=len(flat);e=np.full(l,-1,np.int32);c=np.zeros(l,np.int32);rows=np.full((l,cap),-1,np.int32);inv=np.full(l,-1,np.int32)
    for expert in np.unique(flat):
        where=np.flatnonzero(flat==expert)
        if not 0<=expert<384 or len(where)>cap:continue
        first=where[0];slot=(first%8)*(l//8)+first//8 if build.get('stripe') else first
        e[slot]=expert;c[slot]=len(where);rows[slot,:len(where)]=where;inv[where]=slot*cap+np.arange(len(where))
    return(e,c,rows,inv)
with torch.inference_mode():
    for t in (1,2,3,4):
        ids=torch.zeros((t,8),dtype=torch.int32,device='hpu');sync();graph=torch.hpu.HPUGraph()
        with torch.hpu.graph(graph):meta=torch.ops.gaudi_smallm.metadata(torch.ops.gaudi_smallm.flatten(ids.clone()),4)
        sync()
        base=np.arange(t*8,dtype=np.int32).reshape(t,8)
        mixed=np.array([[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,2,3,12,13,14,15],[4,5,6,7,16,17,18,19]],np.int32)[:t]
        invalid=base.copy();invalid[0,0]=-1;invalid[-1,-1]=384
        states={'uniform':base,'hot':np.tile(np.arange(383,375,-1,dtype=np.int32),(t,1)),'mixed':mixed,'invalid':invalid,'overflow':np.zeros_like(base),'restore':base}
        for name,raw in states.items():
            ids.copy_(torch.from_numpy(raw));sync();graph.replay(asynchronous=True);sync()
            actual=[v.cpu()for v in meta];expected=oracle(raw,4)
            equal=all(np.array_equal(v.numpy(),e)for v,e in zip(actual,expected))
            records.append(dict(T=t,state=name,pass_=equal));torch.save(dict(ids=torch.from_numpy(raw),actual=actual,expected=[torch.from_numpy(v)for v in expected]),out/f't{t}-{name}.pt')
            (out/'result.json').write_text(json.dumps(dict(status='CHECKING',records=records),indent=2)+'\n');assert equal,records[-1]
        del graph,meta,ids;sync()
(out/'result.json').write_text(json.dumps(dict(status='PASS_DEVICE_METADATA_ORACLE',records=records),indent=2)+'\n')
