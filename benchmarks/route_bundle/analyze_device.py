"""Compare final bundle to an already qualified three-database row-first run."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def tensors(data,prefix=''):
    for key,value in data.items():
        if isinstance(value,torch.Tensor):yield prefix+key,value
        elif isinstance(value,dict):yield from tensors(value,prefix+key+'.')

p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();result=json.loads((a.bundle/'result.json').read_text());exit_record=json.loads((a.bundle/'exit.json').read_text())
assert result['status']=='PASS_TP_LOCAL_MOE_FP32_BOUND'
assert result['bundle_binding']['external_TPC_libraries']==1
assert exit_record['runner_exit_code']==0 and exit_record['postflight']['returncode']==0
assert len(exit_record['kernel_libraries_sha256'])==1
audit=json.loads((a.bundle/'audit.json').read_text());assert audit['all_placement_pass']
original_exit=json.loads((a.original/'exit.json').read_text())
assert '22dc3a89af29d441c16e0664dcda9094a22bdcb19affc08f8ef430444eceee46'in original_exit['kernel_libraries_sha256'].values()
rows=[];files={}
for state in ('uniform','hot','skew','zero','invalid','restored'):
    old=a.original/(state+'.pt');new=a.bundle/(state+'.pt')
    files[state]={k:hashlib.sha256(p.read_bytes()).hexdigest()for k,p in [('original',old),('bundle',new)]}
    expected=dict(tensors(torch.load(old,weights_only=False)));actual=dict(tensors(torch.load(new,weights_only=False)))
    assert actual.keys()==expected.keys()
    for name,want in expected.items():
        got=actual[name];assert got.dtype==want.dtype and got.shape==want.shape
        x=want.contiguous().view(torch.uint8);y=got.contiguous().view(torch.uint8)
        bad=int((x!=y).sum());assert bad==0,(state,name,bad)
        rows.append(dict(state=state,tensor=name,elements=got.numel(),dtype=str(got.dtype),
                         shape=list(got.shape),mismatched_bytes=bad,
                         raw_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
report=dict(status='PASS_BUNDLE_WHOLE_GRAPH_AND_ORIGINAL_RAW_BITS',
            original_case=str(a.original),bundle_case=str(a.bundle),comparisons=rows,
            output_FP32_cells=sum(r['elements']for r in rows if r['tensor']in('y','consumer')),
            saved_payload_sha256=files,bundle_binding=result['bundle_binding'],
            event_median_us=result['event_median_us'],wall_median_us=result['wall_median_us'],
            placement=[{k:r[k]for k in ('compute_nodes','decoder_nodes','workspace_bytes',
                                       'decoded_weight_SRAM_address_union_bytes','placement_pass')}
                       for r in audit['records']],
            scope='Single full gate; no performance comparison claim. Original qualified row-first arithmetic unchanged.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('comparisons','saved_payload_sha256')}))
