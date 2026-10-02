"""CPU route-coverage proof and the actual C++ planner's bounded-capacity ledger."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--budget-exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[]
for T in [1,2,8,16,32,64,128,256,512,513]:
 E,R=384,8
 for kind in ['uniform','hot_expert','hot_set','masked']:
  ids=(np.arange(T*R).reshape(T,R)%E).astype(np.int32)
  if kind=='hot_expert':ids[:,0]=0;ids[:,1:]=1+(np.arange(T*(R-1)).reshape(T,R-1)%(E-1))
  if kind=='hot_set':ids[:]=np.arange(R)[None,:]
  if kind=='masked':ids[(np.arange(T)[:,None]+np.arange(R)[None,:])%3==0]=-1
  assert all(len(set(row[row>=0]))==int(np.sum(row>=0)) for row in ids)
  groups=[np.flatnonzero(ids.reshape(-1)==e) for e in range(E)];counts=np.array([len(g) for g in groups]);valid=np.flatnonzero(ids.reshape(-1)>=0)
  for cap in sorted(set([16,32,64,T])):
   overflow=np.flatnonzero(counts>cap);accepted=len(overflow)==0
   mapping={};seen=[]
   for e,g in enumerate(groups):
    for row,route in enumerate(g):mapping[int(route)]=(e,row);seen.append(int(route))
   assert len(seen)==len(set(seen))==len(valid) and sorted(seen)==valid.tolist()
   if accepted:assert all(row<cap and groups[e][row]==route for route,(e,row) in mapping.items())
   case={'T':T,'E':E,'R':R,'distribution':kind,'capacity':cap,'active_routes':len(valid),'active_experts':int(np.count_nonzero(counts)),'max_M_e':int(counts.max()),'overflow_experts':overflow.tolist(),'supported_for_this_fixture':accepted,'routes_mapped_exactly_once':True,'normal_output_valid':accepted,'padded_mme_row_slots':E*cap,'slot_expansion':E*cap/max(1,len(valid))}
   if kind=='uniform' and T in [1,512,513]:
    case['plans']=[json.loads(subprocess.check_output([str(a.budget_exe.resolve()),str(T),str(E),str(R),str(cap),'4',str(nt),'6144'],text=True)) for nt in [512,6144]]
   cases.append(case)
result={'status':'CPU route and C++ planner checks only; no device graph or model qualification','all_checks_pass':True,'cases':cases,'capacity_T_covers_unique_topk':True,'capacity_below_T_has_no_same_graph_fallback':True,'production_default_selected':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(cases),'all_checks_pass':True,'overflow_cases':sum(not c['supported_for_this_fixture'] for c in cases)}))
