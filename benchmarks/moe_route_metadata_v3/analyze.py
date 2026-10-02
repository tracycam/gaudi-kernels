"""Complete physical profile schedule + paired metadata-only summaries."""
import argparse,hashlib,json,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=a.directory;d=json.loads((root/'results/paired-a/result.json').read_text());assert d['status']=='PASS_METADATA_PAIRED'
paired=[]
for state in ('uniform','hot','skew'):
    r={'state':state}
    for v in ('v2','v3'):
        samples=[x for x in d['timings']if x['state']==state and x['variant']==v];assert len(samples)==4
        r[v]={k:statistics.median(x[k]for x in samples)for k in ('event_us','wall_us')};r[v]['all_windows']=samples
    paired.append(r)
case=root/'results/paired-profile-b';pd=json.loads((case/'result.json').read_text());assert pd['status']=='PASS_METADATA_PAIRED'
path=case/'post_graph.json';files=list(path.rglob('*.json'))if path.is_dir()else[path];mapping={};node_details={}
for f in files:
 for g in json.loads(f.read_text())['graphs']:
  for v in ('v2','v3'):
   if any(n['guid']=='gk_route_count_i32_'+v for n in g['nodes']):
    assert v not in mapping.values();mapping[g['name']]=v;node_details[g['name']]={int(n['id']):n for n in g['nodes']if not n['is_logical']}
assert set(mapping.values())=={'v2','v3'}
trace=list((case/'trace').glob('*_000-300.json'));assert len(trace)==1
raw=analyze(trace[0],path,list(mapping));summaries=[]
for name,v in mapping.items():
 inv=sorted((r for r in raw['invocations']if r['recipe']==name),key=lambda r:r['start_us']);windows=[r for r in pd['timings']if r['variant']==v]
 expected=1+6+3+sum(r['replays']for r in windows);assert len(inv)==expected,(v,len(inv),expected);at=7
 for state in ('uniform','hot','skew'):
  at+=1;count=sum(r['replays']for r in windows if r['state']==state);selected=inv[at:at+count];at+=count;metrics=[];cores=defaultdict(set)
  for r in selected:
   groups=defaultdict(list)
   for node in r['nodes']:
    groups[node['op']].append(node['envelope_us']);cores[node['op']].add(node['engine_count'])
   metrics.append({'recipe_envelope_us':r['envelope_us'],**{op:sum(xs)for op,xs in groups.items()}})
  witness=[]
  for node in selected[0]['nodes']:
   if not node['op'].startswith('gk_route_'):continue
   parts=defaultdict(list)
   for pid,tid,begin,end in node['engine_intervals']:parts[(pid,tid)].append((begin,end))
   witness.append(dict(op=node['op'],recipe_idx=selected[0]['recipe_idx'],per_engine_envelope_us=sorted([dict(pid=pid,tid=tid,us=max(e for s,e in xs)-min(s for s,e in xs))for (pid,tid),xs in parts.items()],key=lambda x:-x['us'])))
  summaries.append(dict(variant=v,state=state,invocations=len(selected),physical_nodes=len(node_details[name]),first_invocation_witness=witness,median={k:statistics.median(r[k]for r in metrics)for k in metrics[0]},active_engine_counts={k:sorted(v)for k,v in cores.items()}))
 assert at==len(inv)
report=dict(status='PASS_METADATA_V3_PROFILE_AND_ABBA',paired=paired,profile=summaries,source_commit=json.loads((root/'source-identity.json').read_text())['git_commit'],scope='Same nine fields plus identical consumers. Profile stage envelopes may overlap, do not sum into latency. No MME/model gain, physical bandwidth or clock utilization claim.',trace_sha256=raw['trace_sha256'])
(a.output/'profile-full.json').write_text(json.dumps(raw,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
