"""Summarize matched complete-recipe slopes; profile events only count engines."""
import argparse
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser();p.add_argument('measurements',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=json.loads(a.measurements.read_text())['records'];by={r['case']:r for r in records}
groups=[]
for tasks,unroll,il in [(n,8,0) for n in (1,6,12,24,48)]+[(24,8,1),(24,32,0)]:
    arms=[by[f'write-t{tasks}-u{unroll}-i{il}-{i}'] for i in range(4)]
    assert all(x['pass_'] and x['full_output_oracle_pass'] and 'profile' not in x for x in arms)
    assert [x['plan']['repeats'] for x in arms]==[128,256,256,128]
    us=[x['event_median_us'] for x in arms]
    delta_us=(us[1]+us[2]-us[0]-us[3])/2
    delta_bytes=arms[1]['plan']['tpc_write_bytes']-arms[0]['plan']['tpc_write_bytes']
    assert delta_us>0 and delta_bytes==tasks*512*256*128
    lanes=by[f'profile-t{tasks}']['profile']['main_lane_event_counts']['TPC']
    groups.append(dict(tasks=tasks,write_unroll=unroll,interleaved=bool(il),arms=[x['case'] for x in arms],
                       event_medians_us=us,delta_us=delta_us,delta_bytes=delta_bytes,
                       incremental_write_TBps=delta_bytes/delta_us/1e6,
                       observed_tpc_lanes=len(lanes),lane_names=list(lanes)))
out=dict(scope='Synthetic TPC store requests; not a physical SRAM bus counter or MXFP4 rate',
         method='ABBA work-volume slope from complete synEvent times, including output drain',
         records=groups,
         conclusion='6/12/24 tasks scale across D0, D0-D1, D0-D3. 48 tasks, interleaving and 32-store unroll do not raise the measured 2.68 TB/s. Single-core rate is higher than aggregate/24: do not assert a universal per-core 64-byte/cycle physical ceiling.',
         actual_mxfp4_target_reached=False)
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(groups,indent=2))
