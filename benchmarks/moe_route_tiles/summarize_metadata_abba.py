"""Isolate metadata v2/v3 in fresh processes with identical full-MoE owners."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.archive/'remote-sha256.json').read_text());assert manifest['status']=='REMOTE_LOCAL_SHA_VERIFIED'
rows=[];saved=[];words=0
for arm,version in enumerate((2,3,3,2)):
    case=a.archive/'results'/f'whole-e384-v{version}-abba-{arm}'
    d=json.loads((case/'result.json').read_text());end=json.loads((case/'exit.json').read_text());audit=json.loads((case/'audit.json').read_text())
    assert end['runner_exit_code']==0 and end['postflight']['returncode']==0
    assert d['status']=='PASS_TP_LOCAL_MOE_FP32_BOUND' and d['metadata_version']==version
    assert audit['all_placement_pass'] and len(audit['records'])==1
    assert (d['fixture']['E'],d['fixture']['T'],d['fixture']['R'],d['C'],d['B'])==(384,513,8,32,500)
    assert all(c['metadata_equal'] and (c.get('bad',0)==0) for c in d['checks'])
    output={}
    for check in d['checks']:
        f=case/(check['state']+'.pt');rel=str(f.relative_to(a.archive));assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest['files'][rel]['sha256']
        # Owned device fixture, hash-verified against the completed local seal.
        # Target Torch serialized protocol4, unsupported by local weights_only.
        output[check['state']]=torch.load(f,weights_only=False)
    saved.append(output)
    rows.append(dict(arm=arm,version=version,event_median_us=d['event_median_us'],wall_median_us=d['wall_median_us'],samples=d['timing'],compute_nodes=audit['records'][0]['compute_nodes'],workspace_bytes=audit['records'][0]['workspace_bytes']))
for other in saved[1:]:
    assert other.keys()==saved[0].keys()
    for state,x in saved[0].items():
        y=other[state]
        for name in ('y','consumer','ids','routing'):
            assert torch.equal(x[name].contiguous().view(torch.uint8),y[name].contiguous().view(torch.uint8)),(state,name)
            if name in ('y','consumer'):words+=x[name].numel()
        common=set(x['metadata'])&set(y['metadata']);assert len(common)>=9
        for name in common:assert torch.equal(x['metadata'][name],y['metadata'][name]),(state,name)
means={v:{k:statistics.mean(r[k]for r in rows if r['version']==v)for k in ('event_median_us','wall_median_us')}for v in (2,3)}
result=dict(status='PASS_SAME_FULL_MOE_METADATA_ABBA',source_commit=manifest['source_commit'],arms=rows,mean_of_arm_medians=means,output_word_pairs=words,output_bits_equal=True,
    event_saved_us=means[2]['event_median_us']-means[3]['event_median_us'],all_decoded_weights_transient_SRAM=True,physical_HBM_measured=False,
    manifest_sha256=hashlib.sha256((a.archive/'remote-sha256.json').read_bytes()).hexdigest(),
    scope='Synthetic T513/E384/top8 full TP-local MoE; same decoder/consumer/core, metadata only differs. Timing includes host graph supply. No model or production promotion.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='arms'},indent=2))
