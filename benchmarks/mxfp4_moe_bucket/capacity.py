"""Record the actual C++ bucket planner's memory and static MME costs."""
import argparse, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--budget-exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[]
for T,E,R in [(8,2,1),(64,6,2)]+[(t,384,8) for t in [1,2,8,16,32,64,128,256,512,513,1024]]:
 for depth in [1,2]:
  b=json.loads(subprocess.check_output([str(a.budget_exe.resolve()),str(T),str(E),str(R),str(depth)],text=True))
  assert b['sram_bytes']<=16*1024**2 and (depth==1 or b['per_buffer_sram_bytes']<=8*1024**2)
  tile_rows=0
  lo,hi=1,16
  while lo<=T:
   slots=min(E,T*R//lo);tile_rows+=slots*((min(hi,T)+127)//128)*128;lo,hi=hi+1,hi*2
  cases.append(dict(T=T,E=E,R=R,scratch_buffers=depth,cap_T_row_slots=E*T,logical_expansion=b['row_slots']/(T*R),illustrative_128row_slots=tile_rows,decoded_sram_written_bytes=b['expert_slots']*512*6144*2,all_original_weight_scale_bytes=E*512*6144*17//32,**b))
result={'scope':'actual C++ graph planner budgets, not measured MME work/performance','device_overlap_proved':False,'cases':cases};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(cases),'sram_limits_checked':True}))
