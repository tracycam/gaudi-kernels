"""Compare canonical residual sums, not just the two deferred-add tensors."""
import argparse,json,re
from pathlib import Path
import torch


def compare(a,b):
    a,b=a.float(),b.float();d=a-b
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/b.norm().clamp_min(1e-30)),
                rms=float(d.square().mean().sqrt()),bf16_equal=torch.equal(a.bfloat16(),b.bfloat16()))


def analyze(root,trusted_local_pickle=False):
    records=[]
    for metadata in sorted(root.glob('*-mtp.json')):
        for audit in json.loads(metadata.read_text()).get('audit',[]):
            if not audit.get('stages'):continue
            path=root/'replay-audit'/Path(audit['path']).with_suffix('.layers.pt').name
            data=torch.load(path,map_location='cpu',weights_only=not trusted_local_pickle)
            names=sorted({k.rsplit('/',1)[0]for k in data},key=lambda k:int(re.search(r'layers\.(\d+)',k)[1]))
            layers=[]
            for name in names:
                assert name+'/0'in data and name+'/1'in data
                h,r=data[name+'/0'],data[name+'/1']
                layers.append(dict(layer=int(re.search(r'layers\.(\d+)',name)[1]),
                    hidden=compare(h['batched'],h['single']),residual=compare(r['batched'],r['single']),
                    fp32_sum=compare(h['batched'].float()+r['batched'].float(),h['single'].float()+r['single'].float())))
            records.append(dict(phase=metadata.stem,case=audit['case'],checks=audit['checks'],layers=layers,
                first_sum_bf16_difference=next((r['layer']for r in layers if not r['fp32_sum']['bf16_equal']),None)))
    return dict(scope='Same-prefix layer localization. Differences alone do not classify a bug or legal FP32 rounding.',records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trusted-local-pickle',action='store_true',help='Only for our owned replay-audit tensors saved by the frozen SDK pickle4 bridge')
    a=p.parse_args()
    result=analyze(a.case,a.trusted_local_pickle);a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['records']:
        print(r['phase'],r['case'],'first BF16 residual-sum difference',r['first_sum_bf16_difference'])
        for layer in r['layers']:
            if layer['layer']<12 or layer['layer']%10==9:print(layer['layer'],layer['fp32_sum'])
