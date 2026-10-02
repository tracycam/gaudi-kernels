"""Actual native engine intervals; complete three-node chain, no host subtraction."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
p=argparse.ArgumentParser();p.add_argument('--gate',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
gate=json.loads((a.gate/'result.json').read_text());profile=json.loads((a.profile/'result.json').read_text())
assert gate['status']==profile['status']=='PASS_FULL_ATTENTION' and len(gate['records'])==14
for record in gate['records']:
 assert set(record['variants'])=={'quad','group','quad_debug','group_debug'}
 for name,variant in record['variants'].items():
  assert variant['finite'] and variant['context_baseline_bits']==variant['consumer_baseline_bits']==0
  assert variant['relative_l2_fp64']<.003
  if name=='group_debug':
   for key in ['scores_baseline_fp32_bits','accumulator_baseline_fp32_bits','normalized_fp32_baseline_fp32_bits']:assert variant[key]==0
post=a.profile/'post_graph.json';files=list(post.rglob('*.json'))if post.is_dir()else[post]
graphs=[g for f in files for g in json.loads(f.read_text())['graphs']]
selected={}
for name in ('quad','group'):
 matches=[g for g in graphs if any(n['guid']==f'gk_swa128_avgroup_{name}_v0'for n in g['nodes'])];assert len(matches)==1
 selected[name]=matches[0]
trace=list((a.profile/'trace').glob('*.json'));assert len(trace)==1
r=analyze(trace[0],post,[g['name']for g in selected.values()]);summary={}
assert [(x['position'],x['kind'])for x in profile['records']]==[(128,'ordinary'),(32767,'ordinary'),(0,'ordinary')]
for variant,g in selected.items():
 rows=sorted((x for x in r['invocations']if x['recipe']==g['name']),key=lambda x:x['start_us']);assert len(rows)==23
 assert all(x['physical_nodes']==3 for x in rows)
 summary[variant]={}
 for index,pos in enumerate([128,32767,0]):
  chosen=rows[2+index*7:2+(index+1)*7];assert len(chosen)==7
  nodes=[n for x in chosen for n in x['nodes']if n['op']==f'gk_swa128_avgroup_{variant}_v0'];assert len(nodes)==7 and all(n['engine_count']==16 for n in nodes)
  summary[variant][str(pos)]={'native_chain_median_us':statistics.median(x['envelope_us']for x in chosen),'attention_TPC_median_us':statistics.median(n['envelope_us']for n in nodes),'invocations':7,'attention_engines':16,'physical_nodes':3}
r.update(summary=summary,gate_cases=14,fp32_words_compared_per_state=14*2048,bf16_words_compared=14*2048,
         max_relative_l2_FP64=max(x['variants']['group']['relative_l2_fp64']for x in gate['records']),
         invocation_assignment='Two capture/warm executions then three position states, each one correctness plus six ABBA invocations, matched by actual recipe names and native IDs',
         unprofiled_abba=gate['abba_sequence'])
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(summary,indent=2))
