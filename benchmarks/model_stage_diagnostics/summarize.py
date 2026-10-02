"""Recompute the existing logit gate and retain the first QKV witness; CPU only."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
result=json.loads((a.case/'result.json').read_text());logits=[]
for record in result['same_contract_reference']:
 if not record['reference_policy'].startswith('cpu_native'):continue
 tags=[record[k].replace('_','-').replace('+','-')+'-teacher-'+str(record['position']) for k in ('reference_policy','policy')]
 files=[a.case/'quality-logits'/(t+'.pt') for t in tags];x,y=[torch.load(path,map_location='cpu',weights_only=False) for path in files]
 assert torch.equal(x['input_ids'],y['input_ids']) and torch.equal(x['positions'],y['positions'])
 ref,cand=x['logits'].double(),y['logits'].double();lp,lq=ref.log_softmax(-1),cand.log_softmax(-1)
 kl=float((lp.exp()*(lp-lq)).sum(-1).max());rel=float((ref-cand).norm()/ref.norm().clamp_min(1e-30));passed=kl<=.01 and rel<=.02
 assert torch.isfinite(ref).all() and torch.isfinite(cand).all()
 assert abs(kl-record['check']['max_kl_ref_to_candidate'])<1e-12 and abs(rel-record['check']['relative_l2'])<1e-12 and passed==record['check']['pass']
 logits.append(dict(reference=record['reference_policy'],candidate=record['policy'],position=record['position'],max_kl=kl,relative_l2=rel,pass_existing_gate=passed,top1_match_fraction=float((ref.argmax(-1)==cand.argmax(-1)).double().mean()),files=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]))
for position in [0,16,48]:
 tags=['decode-a8-bf16-fp32','decode-a8-bf16-fp32-gp-vec','decode-a8-bf16-fp32-gp-vec-down-vec']
 values=[torch.load(a.case/'quality-logits'/f'{t}-teacher-{position}.pt',map_location='cpu',weights_only=False)['logits'] for t in tags]
 assert all(torch.equal(values[0].view(torch.uint8),v.view(torch.uint8)) for v in values[1:])
(a.out/'same-contract-logits.json').write_text(json.dumps(dict(source_result_sha256=hashlib.sha256((a.case/'result.json').read_bytes()).hexdigest(),status=result['status'],candidate_accepted=result['candidate_accepted'],gate=dict(max_kl=.01,relative_l2=.02),gp_down_all_three_offsets_bitwise=True,rows=logits),indent=2)+'\n')
name='model.language_model.model.layers.1.'
files=[a.case/'quality-logits'/f'{tag}-teacher-0-stages-rank6.pt' for tag in ('cpu-native-a8-bf16-fp32','decode-a8-bf16-fp32')]
x,y=[torch.load(path,map_location='cpu',weights_only=False) for path in files]
normalized=x['stages'][name+'input_layernorm'];assert torch.equal(normalized,y['stages'][name+'input_layernorm'])
u,v=x['stages'][name+'self_attn.qkv_proj'],y['stages'][name+'self_attn.qkv_proj'];idx=torch.where(u!=v);assert len(idx[0])==1
record=dict(layer=1,rank=6,output_index=int(idx[1][0]),input_equal=True,reference_value=float(u[idx]),actual_value=float(v[idx]),reference_bits=int(u[idx].view(torch.int16)[0])&65535,actual_bits=int(v[idx].view(torch.int16)[0])&65535)
torch.save(dict(input_bf16=normalized,reference_qkv=u,actual_qkv=v,record=record),a.out/'first-qkv-rank6-layer1.pt');(a.out/'first-qkv-rank6-layer1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
