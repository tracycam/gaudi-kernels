"""Inspect sealed real-model commands/trace; do not infer launches from Python ops."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
p=argparse.ArgumentParser();p.add_argument('--model-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
trace=a.model_run/'native-device-profile-rank0.jsonl';command=a.model_run/'native-phase3-capture0-rank0-commands.txt'
recipes={};sum_starts=collections.Counter();threads=set()
for line in trace.open():
    e=json.loads(line);args=e.get('args',{});rid=args.get('recipe_id');op=args.get('op')
    if e.get('ph')!='B' or not rid or not op:continue
    recipe=recipes.setdefault(rid,{});entry=recipe.setdefault(op,{'first_ts':e['ts'],'starts':0,'names':set()});entry['starts']+=1;entry['names'].add(e['name'])
    if op=='pf_sum_bf16_bf16':sum_starts[rid]+=1;threads.add((e['pid'],e['tid']))
heads=[]
for line in command.open():
    if re.match(r'\d+ kind=',line):heads.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
counts=collections.Counter(v['kind'] for v in heads)
ag=collections.Counter((v['type'],v['count']) for v in heads if v['kind']=='3')
assert counts['1']==147 and counts['2']==69 and counts['3']==73 and ag[('9','6144')]==72
assert len(threads)==24 and sum(sum_starts.values())==72*24*3
selected=[]
for rid,count in sum_starts.items():
    ops=recipes[rid]
    assert len(ops)>1 and any(op.startswith('fused_kernel_') for op in ops),'sum-only recipe'
    selected.append({'recipe_id':rid,'sum_spu_starts':count,'sum_invocations_per_replay':count//(24*3),
                     'ordered_unique_ops':[dict(op=op,**{k:sorted(v) if isinstance(v,set) else v for k,v in entry.items()})
                       for op,entry in sorted(ops.items(),key=lambda item:item[1]['first_ts'])]})
result={'status':'EXISTING_AG_CONSUMER_RECIPE_CONFIRMED','source_files':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [trace,command]},
        'scope':'rank0 real70-layer capture; 3 profiled decode replays; counts are API/recipe and SPU records, not hardware doorbells',
        'launches_per_replay':147,'allreduce_per_replay':69,'allgather_per_replay':73,
        'bf16_hidden_allgathers_per_replay':72,'other_logits_allgather_per_replay':1,'sum_spu_starts':sum(sum_starts.values()),
        'sum_cores_observed':len(threads),'sum_consumer_recipes':selected,
        'observed':'Each BF16 sum is in a larger compute recipe containing its consumers; no separate sum launch',
        'norm_identification_limit':'Fused dynamic GUIDs have no body/ELF in this trace. Source path places norm between sum and QKV/router. Do not claim exact norm ISA from opaque GUID.',
        'fp32_gather_prediction':'No post-sum mark_step in precision_runtime; expected same147 launch structure, but actual FP32 gather capture is unmeasured'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','sum_consumer_recipes']}))
