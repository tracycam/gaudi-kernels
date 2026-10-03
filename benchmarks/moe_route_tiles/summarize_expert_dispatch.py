"""Summarize paired full-chain timings without promoting a best-case default."""
import argparse,json,statistics,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[];cases=[]
for f in sorted(a.results.glob('*/result.json')):
 d=json.loads(f.read_text());cases.append(dict(case=f.parent.name,status=d['status'],error=d.get('error')))
 if not d['status'].startswith('passed_'):continue
 groups=collections.defaultdict(list)
 for x in d['timing']:groups[x['state'],x['pair'],x['trial']].append(x)
 grouped=collections.defaultdict(list)
 for (state,pair,trial),values in groups.items():
  assert sorted(x['arm'] for x in values)==[0,1,2,3]
  base=statistics.mean(x['event_us'] for x in values if x['variant']=='broadcast')
  cand=statistics.mean(x['event_us'] for x in values if x['variant']==pair)
  grouped[state,pair].append(dict(trial=trial,baseline_us=base,candidate_us=cand,ratio=cand/base))
 for (state,pair),trials in grouped.items():
  check=next(x for x in d['checks'] if x['state']==state and x['variant']==pair)
  rows.append(dict(case=f.parent.name,tokens=d['tokens'],state=state,variant=pair,active_experts=check['active_experts'],max_expert_m=check['max_expert_m'],baseline_us=statistics.median(x['baseline_us'] for x in trials),candidate_us=statistics.median(x['candidate_us'] for x in trials),candidate_over_baseline=statistics.median(x['ratio'] for x in trials),trials=trials,relative_l2_to_broadcast=check.get('baseline_relative_l2'),plan=d['plans'][pair]))
report=dict(scope='Single-module complete producer/MoE/consumer HPU Graph event time. Fixed hot/cold replay; no physical HBM-counter claim, model TPS or universal dispatch promotion.',cases=cases,measurements=rows)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(rows),'paired measurements')
