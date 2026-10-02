"""Private same-input prefill stage export; no timing or model acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

p=argparse.ArgumentParser(description=__doc__)
for key in ('input','runtime-fixture','core-library','metadata-library','tiles-library','reference-dir'):
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--input-sha256',required=True)
a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.input)==a.input_sha256=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
assert os.environ['HABANA_VISIBLE_MODULES']==os.environ['GAUDI_KERNELS_MODULE_ID']=='7'
out=Path(os.environ['PROBE_OUT'])
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0',UNIFIED_PRECISION_ROUTER='1',
    GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0',
    ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(2)
runtime=a.runtime_fixture.resolve();plan=json.loads((runtime/'plan.json').read_text())
assert plan['format']=='gk-moe-compact8-runtime-v1'
for name,row in plan['files'].items():assert sha(runtime/name)==row['sha256'],name
ref_manifest=json.loads((a.reference_dir/'manifest.json').read_text())
for name,digest in ref_manifest.items():assert sha(a.reference_dir/name)==digest,name
sys.path.insert(0,str(runtime/'executor'));import native_ops,batch_ops,precision_ops
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
for path in (a.core_library,a.metadata_library,a.tiles_library):torch.ops.load_library(str(path.resolve()))
cpu=torch.load(a.input,map_location='cpu',weights_only=False)
assert cpu['x'].shape==(512,6144) and cpu['ids'].shape==(512,8)
def sync():hc.mark_step();torch.hpu.synchronize()
def same(x,y):return x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
result=dict(status='RUNNING',input_sha256=a.input_sha256,cases=[],model_quality_qualified=False,
            performance_qualified=False,scope='Original frozen rank0 input; debug stages must reproduce prior uninstrumented complete output bits; no FP64 reference')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
save()
with torch.inference_mode():
    inputs={k:v.to('hpu')for k,v in cpu.items()};sync()
    for mode in ('broadcast','routed_pure'):
        captures=[];gates=[]
        core,tile=torch.ops.gaudi_moe_reference,torch.ops.gaudi_route_tiles
        actual_mm,actual_gate=core.batch_mm,tile.gate
        def mm(x,w):
            value=actual_mm(x,w)
            if x.shape[-1]==6144:
                assert value.shape[-1]==512
                captures.append(value)
            return value
        def gate(*args):
            value=actual_gate(*args);gates.append(value);return value
        graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
        with patch.object(core,'batch_mm',mm),patch.object(tile,'gate',gate):
            with torch.hpu.graph(graph,stream=stream):
                x=inputs['x']+0
                if mode=='broadcast':
                    y,gp,g,_=batch_ops.moe(x,inputs['ids'],inputs['routing'],inputs['gp'],inputs['gs'],inputs['dp'],inputs['ds'],inputs['table'],inputs['directions'],mode='broadcast',debug=True)
                    exports=dict(gp=gp+0,gate=g+0)
                else:
                    y,meta=moe_route_tiles(x,inputs['ids'],inputs['routing'],inputs['gp'],inputs['gs'],inputs['dp'],inputs['ds'],inputs['table'],layout=1,rows=32,n_tile=2048,metadata_version=3)
                    assert len(captures)==len(gates)==(meta.capacity+1)//2
                    exports=dict(gp=torch.cat(captures,dim=0)+0,gate=torch.cat(gates,dim=0)+0,
                                 inverse=meta.inverse+0,valid_rows=meta.valid_rows+0,tile_expert=meta.tile_expert+0,status=meta.status+0)
                consumer=y+.03125
        sync();graph.replay(asynchronous=True);sync()
        data=dict(y=y.cpu(),consumer=consumer.cpu(),**{k:v.cpu()for k,v in exports.items()})
        path=out/(mode+'-stages.pt');torch.save(data,path,pickle_protocol=4)
        ref=torch.load(a.reference_dir/(mode+'.pt'),weights_only=False,map_location='cpu')
        relation={k:same(data[k],ref[k])for k in ('y','consumer')}
        row=dict(mode=mode,output_file=path.name,output_sha256=sha(path),plain_output_bits_equal=relation,
                 stages={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in data.items()})
        result['cases'].append(row);save()
        assert all(relation.values()),'Debug graph changed complete arithmetic; do not attribute its stages to plain path'
        if mode=='routed_pure':assert data['status'].tolist()==[0]
    result['status']='DIAGNOSTIC_PREFILL_STAGES_WITH_COMPLETE_OUTPUT_RELATION';save()
