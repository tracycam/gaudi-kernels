"""E384 production shard costs, not a latency predictor or physical HBM audit."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_moe_plan import MoePlan
records=[]
for T in [2,8,64,512]:
 for mode in ['capacity_t','bucket']:
  for distribution,active,valid in [('uniform',min(384,T*8),T*8),('hot8',8,T*8),('ragged_half',min(384,T*4),T*4)]:
   d=MoePlan(mode=mode).costs(T,8,384,active);d.update(distribution=distribution,valid_routes=valid,padding_rows_per_useful_route=d['row_slots']/valid,logical_source_read_amplification=1,physical_HBM_measured=False,MME_minrow_internal_padding_measured=False)
   # Each expert belongs to exactly one bucket; empty slots skip W/S loads.
   # The 1x claim assumes the selected certified-fast path, not exact fallback.
   records.append(d)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(records=records,latency_predicted=False,device_validated=False),indent=2)+'\n')
