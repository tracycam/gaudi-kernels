"""Explain saved first attention differences and the layer17 route-set switch."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--bias',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
case=json.loads((a.case/'result.json').read_text());assert case['plain_staged_relation_status']=='COMPLETE'
bias_record=json.loads(a.bias.read_text());dtype=getattr(torch,bias_record['dtype'].removeprefix('torch.'))
bias=torch.tensor(bias_record['values'],dtype=dtype)
assert list(bias.shape)==bias_record['shape'] and hashlib.sha256(bias.view(torch.uint8).numpy().tobytes()).hexdigest()==bias_record['tensor_bytes_sha256']
assert bias_record['key']=='model.layers.17.mlp.gate.e_score_correction_bias'
stage_rows=[r for r in case['rows']if r['staged']];ranks=[];route_checks=[]
def ordered(x):
    u=x.view(torch.int16).to(torch.int32)&65535
    return torch.where(u&32768!=0,65535-u,u+32768)
for rank in range(8):
    states=[]
    for row in stage_rows:
        record=next(f for f in row['stage_files']if f['rank']==rank);path=a.case/'quality-logits'/Path(record['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        states.append(torch.load(path,weights_only=False)['stages'])
    key='model.language_model.model.layers.7.self_attn.attn';x,y=[s[key]for s in states]
    assert x.dtype==y.dtype==torch.bfloat16
    delta=(ordered(x)-ordered(y)).abs();mask=delta!=0
    ranks.append(dict(rank=rank,changed_values=int(mask.sum()),total=x.numel(),
                      max_bf16_ulp=int(delta.max()),max_abs=float((y.float()-x.float()).abs().max())))
    prefix='model.language_model.model.layers.17.mlp.'
    for name,state in zip(('old','routed'),states):
        raw=state[prefix+'gate'].float();score=raw.sigmoid()+bias.float()
        ids=state[prefix+'experts.routed_experts.moe_op.route_ids']
        _,selected=score.topk(8,dim=-1)
        matched=torch.equal(selected.sort(-1).values,ids.to(torch.int64).sort(-1).values)
        assert matched,'CPU FP32 score does not reproduce saved route set'
        top=score.topk(10,dim=-1)
        route_checks.append(dict(rank=rank,path=name,cpu_fp32_top8_set_matches_saved=True,
            expert6_minus99=float(score[0,6]-score[0,99]),
            top10_ids=top.indices[0].tolist(),top10_scores=top.values[0].tolist(),
            saved_ordered_ids=ids.tolist()))
report=dict(attention_layer=7,attention=ranks,route_layer=17,route_checks=route_checks,
    changed_attention_values=sum(r['changed_values']for r in ranks),
    max_attention_bf16_ulp=max(r['max_bf16_ulp']for r in ranks),
    checkpoint_bias={k:v for k,v in bias_record.items()if k!='values'},
    bias_json_sha256=hashlib.sha256(a.bias.read_bytes()).hexdigest(),
    scope='CPU FP32 sigmoid+bias re-evaluation reproduces both saved route sets. This isolates a real changed-score decision; it does not bound every prefill operation or prove the initial KV difference is unavoidable rounding. No FP64 model reference or acceptance override.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(changed_attention_values=report['changed_attention_values'],max_bf16_ulp=report['max_attention_bf16_ulp'],first_rank_route=route_checks[:2])))
