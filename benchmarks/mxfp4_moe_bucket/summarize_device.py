"""Audit retrieved native buckets; keep profiled timings and placement separate."""
import argparse,csv,json,math,re,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[]
for path in sorted(a.results.iterdir()):
 if not (path/'exit.json').exists():continue
 status=json.loads((path/'exit.json').read_text());assert status['runner_exit_code']==0 and status['source_identity_verified'];f=json.loads((path/'fixture.json').read_text());assert not np.any(np.fromfile(path/'overflow.bin',np.int32))
 rows=[json.loads(line) for line in (path/'run.log').read_text().splitlines() if line.startswith('{')];times=[r for r in rows if r.get('stage')=='timing'];gp=f['mode']=='gate'
 if gp:
  got=np.fromfile(path/'output.bin',np.uint16);ref=np.fromfile(path/'reference_gate.bin',np.uint16);bad=int(np.count_nonzero(got!=ref));maxerr=0
 else:
  got=np.fromfile(path/'output.bin',np.float32);ref=np.fromfile(path/'reference.f64',np.float64);norm=np.fromfile(path/'sumabs.f64',np.float64);error=np.abs(got.astype(np.float64)-ref);bad=int(np.count_nonzero(~np.isfinite(got)|(error>2e-6*norm+1e-35)));maxerr=float(error.max())
 assert not bad
 post=path/'post_graph.json';post=next(post.glob('*.json')) if post.is_dir() else post;g=json.loads(post.read_text())['graphs'][0];decoded=[t for t in g['tensors'] if t['name'].endswith('_decoded_sram')]
 assert decoded and all(t['allocation']=='SRAM' and t['rmw_section'] and not t['persistent'] for t in decoded)
 regions={}
 for t in decoded:regions[t['offset']]=max(regions.get(t['offset'],0),math.prod(t['max_shape'])*2)
 rec={'case':path.name,'commit':status['git_commit'],'kernel_libraries_sha256':status['kernel_libraries_sha256'],'checked':int(got.size),'bad':bad,'max_abs':maxerr,'fixture':{k:f[k] for k in ['N','K','T','R','E','M_e','weight_layout','active_routes','active_weight_scale_bytes','all_weight_scale_bytes']},'profiled':'-profile-' in path.name,'event_us_median':statistics.median(t['event_us'] for t in times) if times else None,'wall_us_median':statistics.median(t['wall_us'] for t in times) if times else None,'workspace_bytes':g['workspace_size'],'decoded_physical_sram_regions_bytes':sorted(regions.values()),'compute_nodes':len([n for n in g['nodes'] if not n['is_logical']]),'MME_nodes':len([n for n in g['nodes'] if n.get('engine')=='MME']),'physical_HBM_counters_measured':False}
 tensors={t['name']:t for t in g['tensors']};mme_nodes=[n for n in g['nodes'] if n.get('engine')=='MME'];assert all(tensors[n['input_tensors'][1]]['allocation']=='SRAM' for n in mme_nodes)
 rec['requested_MME_FLOPs_before_hardware_padding']=sum(2*math.prod(tensors[n['output_tensors'][0]]['max_shape'])*tensors[n['input_tensors'][0]]['max_shape'][0] for n in mme_nodes)
 if rec['profiled']:
  events=list(csv.DictReader(next((path/'trace').glob('*analyzed_nodes.csv')).open()));dec=[n for n in events if n['Node Name'].endswith('_decode')];mme=[n for n in events if n['Unit']=='MME'];overlap=[]
  for d in dec:
   for m in mme:
    dt=max(0,min(float(d['End time of node']),float(m['End time of node']))-max(float(d['Start time of node']),float(m['Start time of node'])))
    if dt:overlap.append({'decode':d['Node Name'],'MME':m['Node Name'],'overlap_us':dt})
  rec['trace']={'decode_start_slot_order':[int(re.search(r'slot(\d+)',d['Node Name']).group(1)) for d in sorted(dec,key=lambda n:float(n['Start time of node']))],'decode_MME_overlap':overlap,'decode_duration_sum_us':sum(float(n['Duration (us)']) for n in dec),'MME_duration_sum_us':sum(float(n['Duration (us)']) for n in mme),'metadata_and_epilogue':[{k:n[k] for k in ['Node Name','Duration (us)','Input placement','Output placement']} for n in events if n not in dec and n not in mme]}
 cases.append(rec)
baseline=a.results/'capt-t64-historical-v1'/'output.bin';baseline_data=baseline.read_bytes()
assert all((a.results/n/'output.bin').read_bytes()==baseline_data for n in ['bucket-t64-depth1-historical-v1','bucket-t64-depth2-historical-v1','bucket-t64-depth1-native-v1','bucket-t64-depth2-native-v1'])
summary={'all_attempts_successful':False,'pre_acquire_build_failures':[{'source_commit':'7a57544','reason':'formal build invoked unavailable git before reading source identity','fixed_in':'f62e5a7','device_acquired':False}],'all_device_quality_checks_pass':True,'checked':sum(c['checked'] for c in cases),'T64_same_logical_inputs_all_unprofiled_outputs_bitwise_equal':True,'bucket_promoted':False,'dual_buffer_promoted':False,'production_E384_device_tested':False,'cases':cases,'limits':['E6/T64 is a cross-bucket correctness/trace case, not representative E384 routing performance','Historical single resident layout is accepted without weight copy or repack','Double buffer executes odd chain then even chain; isolated boundary overlap is not a steady pipeline','No full-model or physical HBM bandwidth claim']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'cases':len(cases),'checked':summary['checked'],'all_device_quality_checks_pass':True,'promotion':False}))
