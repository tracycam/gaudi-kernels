"""Summarize real public-graph gates and profiler intervals without hiding SRAM failure."""
import argparse,collections,csv,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
all_case=a.device/'results/neumaier-hand-public-all-a';profile_case=a.device/'results/neumaier-hand-public-profile-a'
run=json.loads((all_case/'public/summary.json').read_text());placement=json.loads((all_case/'public/placement-audit.json').read_text());profile_run=json.loads((profile_case/'public/summary.json').read_text())
for case in [all_case,profile_case]:
 exit=json.loads((case/'exit.json').read_text());assert exit['runner_exit_code']==0 and exit['postflight']['returncode']==0 and exit['postflight']['cleanup']['reaped']
 assert any(line.split(',')[0].strip()=='7' and '768 MiB' in line and '0 %' in line for line in (case/'device-after.txt').read_text().splitlines())
assert run['all_pass'] and profile_run['all_pass']
assert [r['variant'] for r in profile_run['trace_replays']]==['old','hand','hand','old']
assert all(r['repeats']==r['public_launches']==2 for r in profile_run['trace_replays'])
path=next(a.device.rglob('*analyzed_nodes.csv'));nodes=list(csv.DictReader(path.open()));guids={'old':'gk_block128_reduce_neumaier_row1_v1','hand':'gk_neumaier_isa_handschedule_v1'};actual={};last_rows=[]
for variant,guid in guids.items():
 rows=sorted((r for r in nodes if r['Op Type']==guid),key=lambda r:float(r['Start time of node']));assert len(rows)==8
 tail=rows[-4:];last_rows += [(float(r['Start time of node']),variant) for r in tail]
 groups=collections.defaultdict(list)
 for row in nodes:
  if row['Graph Name']==rows[0]['Graph Name']:groups[row['Unique node id']].append(row)
 assert all(len(rs)==8 for rs in groups.values())
 ordered=[sorted(rs,key=lambda r:float(r['Start time of node'])) for rs in groups.values()]
 envelopes=[max(float(rs[i]['End time of node']) for rs in ordered)-min(float(rs[i]['Start time of node']) for rs in ordered) for i in range(8)]
 durations=[float(r['Duration (us)']) for r in tail]
 actual[variant]=dict(guid=guid,all_reducer_node_durations_us=[float(r['Duration (us)']) for r in rows],abba_reducer_node_durations_us=durations,abba_reducer_node_median_us=statistics.median(durations),abba_recipe_node_envelope_us=envelopes[-4:],abba_recipe_node_envelope_median_us=statistics.median(envelopes[-4:]),physical_nodes_per_recipe=dict(collections.Counter(rs[0]['Op Type'] for rs in ordered)),profiled_replays=len(rows),input_placement=tail[0]['Input placement'])
assert [v for _,v in sorted(last_rows)]==['old','old','hand','hand','hand','hand','old','old']
bridge=collections.defaultdict(list)
for row in run['timings']:
 assert row['valid'] and row['public_launches']==row['repeats'];bridge[(row['case'],row['variant'])].append(row)
bridge_rows=[dict(case=case,variant=variant,median_event_us=statistics.median(r['event_us'] for r in rows),median_wall_us=statistics.median(r['wall_us'] for r in rows),event_us=[r['event_us'] for r in rows],wall_us=[r['wall_us'] for r in rows]) for (case,variant),rows in bridge.items()]
old=actual['old']['abba_reducer_node_median_us'];new=actual['hand']['abba_reducer_node_median_us']
report=dict(all_requirements_pass=bool(run['all_pass'] and placement['all_pass']),numerics_pass=True,public_single_submit_pass=True,timing_consistency_pass=True,sram_placement_pass=placement['all_pass'],logical_weight_read_ratio_one=all(r['placement']['no_logical_weight_read_amplification'] for r in placement['records']),checked_outputs=sum(r.get('checked',0) for r in run['numerics']),saved_partial_checked_outputs=sum(r.get('checked',0) for r in run['numerics'] if r['phase']=='reduction'),whole_chain_checked_outputs=sum(r.get('checked',0) for r in run['numerics'] if r['phase']=='chain'),profiler=dict(source_csv_sha256=sha(path),actual_node_intervals=actual,reducer_saved_us=old-new,reducer_duration_reduction_fraction=(old-new)/old,spu_vpu_counters_available=False,physical_hbm_counters_measured=False),bridge_inclusive_timings=bridge_rows,placement=placement,target_build=json.loads((a.target/'builds/torch/build.json').read_text()),target_source_and_elf=json.loads((a.target/'builds/verify/source-elf-check.json').read_text()),qualification='Numerical and exact-order ISA improvement over the same DRAM-partial baseline. The SRAM contract fails for both variants. No model quality or TPS acceptance is inferred.',module7_released=True)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['all_requirements_pass','numerics_pass','sram_placement_pass','checked_outputs','saved_partial_checked_outputs','whole_chain_checked_outputs','module7_released']}|dict(reducer_old_us=old,reducer_hand_us=new,reducer_saved_us=old-new),indent=2))
