"""Audit a completed exact substage without changing overall failed-case status."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from summarize import aggregate
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);a=p.parse_args()
exitdata=json.loads((a.case/'exit.json').read_text());assert exitdata['runner_exit_code']!=0
ref=np.fromfile(a.fixtures/'exact/oracle-fp64.bin',np.float64).reshape(69,6144);rows=[];apis=[];checked=0
for rank in range(8):
 d=a.case/'exact';records=[json.loads(v) for v in (d/f'rank{rank}.jsonl').read_text().splitlines()]
 assert len([r for r in records if r['stage']=='quality'])==8
 timing=[r for r in records if r['stage']=='timing']
 assert len(timing)==20 and {(r['phase'],r['trial']) for r in timing}=={(p,t) for p in range(4) for t in range(5)}
 assert all(r['repeats']==10 for r in timing)
 assert all(r['bad']==0 and r['checked']==69*6144 and r['ordered_FP32_bit_differences']==0 for r in records if r['stage']=='quality');rows+=records
 for phase in range(4):
  ag=phase in (1,2);tag=('ags' if ag else 'ar')+str(phase)
  calls=[json.loads(v) for v in (d/f'rank{rank}-{tag}-api.jsonl').read_text().splitlines()]
  comm=[r for r in calls if r['api'].startswith('hccl')];launch=[r for r in calls if r['api'].startswith('synLaunch')]
  assert len(comm)==69 and all(r['count']==6144 and r['dtype']==7 and r['status']==0 and r['api']==('hcclAllGather' if ag else 'hcclAllReduce') and r['reduce_op']==(-1 if ag else 0) for r in comm)
  assert len(launch)==(69 if ag else 0) and all(r['status']==0 for r in launch)
  apis.append(dict(rank=rank,phase=phase,collectives=len(comm),sum_launches=len(launch),dtype=7,count=6144))
  for suffix in ['', '-post']:
   quality=[r for r in records if r.get('tag')==tag+suffix];assert len(quality)==1
   assert quality[0]['gather_bytes_exact']==ag
   got=np.fromfile(d/f'rank{rank}-{tag}{suffix}.bin',np.float32).reshape(69,6144);assert np.array_equal(got.astype(np.float64),ref);assert np.array_equal(got.view(np.uint32),ref.astype(np.float32).view(np.uint32));checked+=got.size
trials=aggregate(rows);phases=[]
for phase in range(4):
 records=[r for r in trials if r['phase']==phase];wall=statistics.median(r['slowest_rank_wall_us_per_69'] for r in records)
 phases.append(dict(phase=phase,mode=records[0]['mode'],wall_us_per_69=wall,event_us_per_69=statistics.median(r['slowest_rank_event_us_per_69'] for r in records),wall_us_per_collective=wall/69,median_last_submit_us_per_69=statistics.median(r['latest_submit_us_per_69'] for r in records)))
report=dict(overall_case_pass=False,runner_reason=exitdata['reason'],source_commit=exitdata['source_commit'],exact_substage_complete=True,exact_full_output_recheck_pass=True,checked=checked,ordinary_device_validated=False,phases=phases,trials=trials,api=apis,production_gain_established=False)
(a.case/'exact-partial-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(overall_case_pass=False,checked=checked,phases=phases),indent=2))
