"""Separate unprofiled public submission timing from matched raw TPC intervals."""
import argparse,collections,hashlib,json,statistics,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
summary={'status':'STARTED','scope':'bounded M1 E384 top8 router fixture, not model acceptance','timing':{},'profile':{},'placement':{}}
def distribution(values):return {'samples':values,'count':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)}
for case in ('capture-a','timing-a','profile-a','timing-factor1-b','profile-factor1-b'):
 exit_record=json.loads((root/case/'exit.json').read_text());result=json.loads((root/case/'result.json').read_text())
 assert exit_record['runner_exit_code']==0 and result['status'].startswith('PASS')
 if case.startswith('timing'):
  pairs={}
  for pair in ('scalar','vector'):
   pairs[pair]={}
   for mode in ('vendor',pair):
    rows=[r for r in result['records'] if r['pair']==pair and r['mode']==mode]
    pairs[pair][mode]={key:distribution([r[key] for r in rows]) for key in ('event_us','wall_us','host_enqueue_us')}
  summary['timing'][case]={'factor':result.get('factor',2.5),'paired':pairs,'profiler_enabled':False}
 if case.startswith('profile'):
  trace=next((root/case/'trace').glob('*.json'));events=json.loads(trace.read_text())['traceEvents']
  ops=sorted({e.get('args',{}).get('op') for e in events if e.get('ph')=='B' and e.get('args',{}).get('HW event name')=='TPC_SPU_START' and e.get('args',{}).get('op') not in (None,'null')})
  core_path=root/case/'core-intervals.json'
  command=[sys.executable,str(Path(__file__).resolve().parents[1]/'moe_graph_views/analyze_core_trace.py'),str(trace),'--ops',*ops,'--output',str(core_path)]
  subprocess.run(command,check=True)
  groups=collections.defaultdict(list)
  for n in json.loads(core_path.read_text())['nodes']:groups[n['recipe']].append(n)
  record={}
  for recipe,ns in groups.items():
   mode='vector' if any('post8_vector' in n['op'] for n in ns) else 'scalar' if any('post8_scalar' in n['op'] for n in ns) else 'vendor'
   counts={len(n['launches']) for n in ns};assert counts=={7},(recipe,counts)
   full=[];post=[];per_op={}
   for n in ns:per_op[n['name']]={'op':n['op'],'active_cores':n['active_cores'],'duration_us':distribution([v['union_us'] for v in n['launches']])}
   for i in range(7):
    intervals=[v for n in ns for v in n['launches'][i]['core_intervals_us'].values()]
    full.append(max(v[1] for v in intervals)-min(v[0] for v in intervals))
    if mode=='vendor':
     gather=next(n for n in ns if n['op']=='gather_elements_fwd_f32');divide=next(n for n in ns if n['op']=='div_precise_f32')
     post.append(max(v[1] for v in divide['launches'][i]['core_intervals_us'].values())-min(v[0] for v in gather['launches'][i]['core_intervals_us'].values()))
    else:post.append(next(n for n in ns if n['op'].startswith('gk_router_post'))['launches'][i]['union_us'])
   record[mode]={'recipe':recipe,'full_TPC_span_us':distribution(full),'post_span_us':distribution(post),'nodes':per_op}
  summary['profile'][case]={'factor':result.get('factor',2.5),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'parser_command':command,'modes':record,
   'interval_scope':'first named TPC SPU start to last named TPC SPU halt; includes intervening gaps, excludes leading/trailing DMA and host; all seven observations retained',
   'post_scope':'vendor gather start through precise-divide halt; candidate single post node; separate factor scaling excluded from vendor post span'}
 if case.startswith('timing'):
  graph=json.loads((root/case/'post_graph.json').read_text())
  record={}
  for g in graph['graphs']:
   nodes=[n for n in g['nodes'] if not n['is_logical']]
   mode='vector' if any('post8_vector' in n['guid'] for n in nodes) else 'scalar' if any('post8_scalar' in n['guid'] for n in nodes) else 'vendor'
   record[mode]={'recipe':g['name'],'physical_engine_nodes':dict(collections.Counter(n['engine'] for n in nodes)),'nodes':[{k:n[k] for k in ('guid','engine','input_tensors','output_tensors')} for n in nodes],
                 'tensors':[{k:t[k] for k in ('name','allocation','max_shape','dtype','persistent')} for t in g['tensors']]}
  summary['placement'][case]=record
capture=json.loads((root/'capture-a/result.json').read_text())
summary['capture']={'records':len(capture['records']),'max_vendor_ulp':max(max(r['max_vendor_ulp']) for r in capture['records']),'max_consumer_ulp':max(max(r['consumer_ulp']) for r in capture['records'])}
summary['status']='DEVICE_FIXTURE_PASS_VECTOR_PREFERRED_MODEL_UNQUALIFIED'
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['status'])
