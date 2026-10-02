"""Read actual old M512/M513 strategies; no runtime import or device calls."""
import argparse,hashlib,json
from pathlib import Path
from contract import PADS,byte_ledger
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
base=a.canonical/'artifacts/builds/block-fp8-native-lanes/module6-a';case=base/'results/native-lanes-profile-b';rows=[];hashes={}
def read(path):
    data=path.read_bytes();hashes[str(path.relative_to(a.canonical))]=hashlib.sha256(data).hexdigest();return json.loads(data)
summary=read(base/'native-profile-analysis.json')['summary']
for path in sorted((case/'post_graph.json').rglob('*.json')):
 for g in read(path)['graphs']:
  ts={t['name']:t for t in g['tensors']};mm=[n for n in g['nodes']if n['engine']=='MME']
  if not mm:continue
  m=ts[mm[0]['input_tensors'][0]]['max_shape'][1]
  if m not in (512,513):continue
  rows.append(dict(M=m,recipe=g['name'],physical_nodes=sum(not n['is_logical']for n in g['nodes']),
      mme=[{k:n[k]for k in ('name','mme_node_strategy','mme_expected_compute_cycles','mme_compute_utilization','num_of_ROIs')}|dict(inputs=[ts[t]['max_shape']for t in n['input_tensors']],outputs=[ts[t]['max_shape']for t in n['output_tensors']])for n in mm],
      profile=next(x for x in summary if x['m']==m)))
old=a.canonical/'artifacts/builds/block-fp8-framework/device-raw/block-final-gates-d'
split=[read(old/name/'result.json')for name in ('qkv-a16-m513-unsplit','qkv-a16-m513-split512')]
report=dict(status='OFFLINE_MEASURED_STRATEGY_AUDIT',source_sha256=hashes,rows=sorted(rows,key=lambda x:x['M']),
    padding_ledger=[byte_ledger(m)for m in PADS],old_split=[dict(case=x['case'],event_us=x['median_event_us'],wall_us=x['median_wall_us'])for x in split],
    interpretation='Observed geometry changes2xh/W256/H512 to2xw/W512/H256; expected cycles+50%, not a clock or measured utilization claim. Padding might retain768 physical rows and save nothing. Old shared-owner split used BF16 scale and was not beneficial.',device_accessed=False)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'rows':len(rows),'old_split':report['old_split']}))
