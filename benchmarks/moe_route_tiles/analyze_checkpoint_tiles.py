"""Recheck sealed real-input row-tile probes and retain all ABBA samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
manifest=json.loads((a.archive/'remote-sha256.json').read_text())
def checked(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][str(path.relative_to(a.archive))]['sha256']
    return path
def read(path):return torch.load(checked(path),map_location='cpu',weights_only=False,mmap=True)
def same(x,y):
    assert x.dtype==y.dtype and x.shape==y.shape
    return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'moe_route_metadata_v3'))
from reference import reference
fixture=read(a.archive/'fixtures/inputs-rank0.pt')
assert hashlib.sha256((a.archive/'fixtures/inputs-rank0.pt').read_bytes()).hexdigest()=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
cases=['checkpoint-c32-abba-0','checkpoint-c16-abba-1','checkpoint-c8-abba-2','checkpoint-c32-abba-3']
words=0;metadata_fields=0;reports=[]
for name in cases:
    folder=a.archive/'results'/name
    exit_record=json.loads(checked(folder/'exit.json').read_text())
    assert exit_record['runner_exit_code']==0 and exit_record['postflight']['returncode']==0
    result=json.loads(checked(folder/'result.json').read_text())
    assert result['status']=='COMPLETE_CHECKPOINT_DIAGNOSTIC_NOT_MODEL_ACCEPTANCE'
    c=result['rows']
    for state in ('original','route_reverse','restored'):
        for mode in ('broadcast','routed_pure','routed_metadata'):
            got=read(folder/f'{state}-{mode}.pt')
            baseline=read(a.archive/'results'/cases[0]/f'{state}-{mode}.pt')
            for key in ('y','consumer'):
                assert bool(got[key].isfinite().all()) and same(got[key],baseline[key]),(name,state,mode,key)
                words+=got[key].numel()
            assert same(got['consumer'],got['y']+.03125)
            if mode=='routed_metadata':
                ids=fixture['ids'].flip(1).contiguous() if state=='route_reverse'else fixture['ids']
                capacity=got['metadata']['tile_expert'].numel()
                expected=reference(ids.tolist(),384,c,capacity)
                for key,value in got['metadata'].items():
                    assert torch.equal(value,torch.tensor(expected[key],dtype=torch.int32)),(name,state,key)
                    metadata_fields+=1
    assert len(result['timing'])==12
    for index,row in enumerate(result['timing']):
        mode=('broadcast','routed_pure','routed_pure','broadcast')[index%4]
        assert (row['trial'],row['arm'],row['mode'])==(index//4,index%4,mode)
        assert row['restored_output_bits_equal'] and row['consumer_bits_equal']
        assert .5<row['event_us']/row['wall_us']<1.2
        path=folder/row['tensor_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        got=read(path);baseline=read(folder/f'restored-{mode}.pt')
        for key in ('y','consumer'):
            assert same(got[key],baseline[key]);words+=got[key].numel()
    placement=a.out/(name+'-placement.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit.py')),str(folder),'--output',str(placement)],check=True)
    layout=json.loads(placement.read_text());assert layout['all_placement_pass']
    reports.append(dict(case=name,rows=c,timing=result['timing'],
        medians={mode:dict(event_us=statistics.median(r['event_us']for r in result['timing']if r['mode']==mode),
                          wall_us=statistics.median(r['wall_us']for r in result['timing']if r['mode']==mode))
                 for mode in ('broadcast','routed_pure')},
        graph_placement=[{k:r[k]for k in ('graph','decoder_nodes','compute_nodes',
            'decoded_weight_SRAM_address_union_bytes','placement_pass')}for r in layout['records']],
        placement_sha256=hashlib.sha256(placement.read_bytes()).hexdigest(),
        cross_algorithm_diagnostics=[r for r in result['checks']if r.get('reference')=='broadcast']))
report=dict(status='PASS_SAME_ALGORITHM_BITS_METADATA_SRAM_TIMING',cases=reports,
            fp32_output_consumer_words=words,metadata_fields_checked=metadata_fields,
            source_input_sha256='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca',
            scope='One actual layer/rank routing distribution; original weights; all timings are full producer/MoE/consumer graph. Not model quality or maximum device utilization. Cross-algorithm error remains diagnostic.',
            model_quality_qualified=False,performance_qualified=False)
(a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],words=words,metadata_fields=metadata_fields,
    rows=[dict(case=r['case'],rows=r['rows'],medians=r['medians'])for r in reports])))
