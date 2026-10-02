"""Recheck saved outputs and summarize unprofiled operator latency separately."""
import argparse,csv,json,statistics,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[];profiles=[]
for case in sorted(a.results.iterdir()):
 if not case.is_dir():continue
 end=json.loads((case/'exit.json').read_text())
 assert end['module']==6 and end['bus_id']=='0000:b3:00.0' and end['runner_exit_code']==0 and end['source_identity_verified']
 records=[]
 for line in (case/'run.log').read_text().splitlines():
  try:records.append(json.loads(line))
  except json.JSONDecodeError:pass
 plan=next(r for r in records if r.get('stage')=='plan');correct=next(r for r in records if r.get('stage')=='correctness')
 assert correct['bad']==0
 controls=next((r for r in records if r.get('stage')=='row_tile'),{'row_n_tile':0,'row_k_tile':32})
 controls.setdefault('row_k_tile',32)
 actual_checked=0
 for i,m in enumerate(map(int,plan['M'].split(','))):
  if not m:continue
  if plan['mode']=='mme':
   output=np.fromfile(case/f'e{i}_output.bin',np.float32).astype(np.float64)
   reference=np.fromfile(case/f'e{i}_oracle_f64.bin',np.float64)
   scale=np.fromfile(case/f'e{i}_sumabs_f64.bin',np.float64)
   assert output.size==reference.size==scale.size==m*plan['N']
   assert np.all(np.isfinite(output)) and np.all(np.abs(output-reference)<=2e-6*scale)
   if plan['pattern']=='zero':assert np.array_equal(output,reference)
  else:
   output=np.fromfile(case/f'e{i}_output.bin',np.uint16);reference=np.fromfile(case/f'e{i}_oracle_bf16.bin',np.uint16)
   assert output.size==reference.size==plan['N']*plan['K'] and np.array_equal(output,reference)
  actual_checked+=output.size
 assert actual_checked==correct['checked']
 if plan['mode']=='mme':
  audit=Path(__file__).with_name('audit_device_graph.py')
  subprocess.run([sys.executable,str(audit),'--case-dir',str(case)],check=True,stdout=subprocess.DEVNULL)
  graph=json.loads((case/'graph-audit.json').read_text())
  assert graph['state']=='compiled_graph_alias_sram_and_element_schedule_pass'
 timed=[r for r in records if r.get('stage')=='timing'];profiled=(case/'profile-command.json').exists()
 row={'case':case.name,'source_commit':end['git_commit'],'profiled':profiled,'plan':plan,'controls':controls,'correctness':correct,
      'saved_output_rechecked':True,'event_us_median':statistics.median(r['event_us'] for r in timed),
      'wall_us_median':statistics.median(r['wall_us'] for r in timed),'event_us_samples':[r['event_us'] for r in timed],
      'event_wall_scope':'same explicit native stream; full graph launch including decode/MME/FP32 epilogue; excludes offline load-time packing and H2D input residency setup'}
 if plan['mode']=='mme':
  row['graph_audit']={'decoded_sram_tiles':len(graph['decoded']),'payload_bytes':sum(v['unique_payload_bytes'] for v in graph['weight_scale_ledger'].values()),
                     'extra_physical_engines':graph['extra_physical_engines'],'mme_geometry_utilization':[v['mme_compute_utilization'] for v in graph['mme_nodes']]}
 if profiled:
  trace=next((case/'trace').glob('*analyzed_nodes.csv'));nodes=list(csv.DictReader(trace.open()));duration={}
  for node in nodes:duration.setdefault(node['Node Name'].split(' iter ')[0],[]).append(float(node['Duration (us)']))
  span=sorted(set(float(n['Bundle duration (us)']) for n in nodes if n['Bundle duration (us)']!='nan'))
  profiles.append({'case':case.name,'scope':'attribution only, excluded from unprofiled latency comparisons',
                   'bundle_span_us':span,'bundle_span_us_median':statistics.median(span),
                   'node_duration_us_median':{k:statistics.median(v) for k,v in duration.items()}})
 rows.append(row)
by_name={r['case']:r for r in rows};comparisons=[]
pairs=[(f'compact-mme-m{m}-n513k257-a',f'compact-mme-m{m}-n513k257-row2048-c') for m in [1,64,257,513]]
pairs +=[(f'compact-extra-random-m{m}-n3392k6144-d',f'compact-wide-m{m}-n3392k6144-g') for m in [1,64]]
pairs +=[('compact-mme-m1-n513k257-row2048-c','compact-wide-m1-n513k257-g')]
for before,after in pairs:
 b,c=by_name[before],by_name[after];assert not b['profiled'] and not c['profiled']
 before_bits=np.fromfile(a.results/before/'e0_output.bin',np.uint32);after_bits=np.fromfile(a.results/after/'e0_output.bin',np.uint32)
 assert before_bits.size==after_bits.size
 comparisons.append({'baseline':before,'candidate':after,'baseline_event_us':b['event_us_median'],'candidate_event_us':c['event_us_median'],
  'latency_reduction_percent':100*(1-c['event_us_median']/b['event_us_median']),
  'saved_fp32_output_bit_mismatches':int(np.count_nonzero(before_bits!=after_bits))})
result={'state':'native_bounded_fixture_gates_pass','module':6,'index':1,'pci':'0000:b3:00.0',
 'native_cases':len(rows),'unprofiled_cases':sum(not r['profiled'] for r in rows),'profiled_cases':len(profiles),
 'unprofiled_mme_output_elements':sum(r['correctness']['checked'] for r in rows if not r['profiled'] and r['plan']['mode']=='mme'),
 'decode_bf16_words':sum(r['correctness']['checked'] for r in rows if r['plan']['mode']!='mme'),
 'numerical_domain':'Device decode: finite E8M0 codes2..252. MME: explicit dyadic fixtures, activation integer/1024, scales118..132, finite FP32 bias integer/256; no generic activation-dependent FTZ certificate.',
 'physical_hbm_transactions':'not measured; emitted graph aliases/SRAM plus requested original byte intervals checked',
 'cases':rows,'comparisons':comparisons,'profiles':profiles}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','comparisons','profiles']},indent=2))
