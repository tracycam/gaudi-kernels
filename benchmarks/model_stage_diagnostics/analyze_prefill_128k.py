"""Audit completed full70 128K prompt scope and physical chunk traces offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--executor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.executor/'executor'))
from analyze_profile import read,duration
result=json.loads((a.case/'result.json').read_text());assert result['context']==131072
assert result['config']['tensor_parallel_size']==8 and not result['config']['enable_prefix_caching']
assert not result['config'].get('hf_overrides') and result['config']['max_num_batched_tokens']==512
prompt=json.loads((a.case/'prompt.json').read_text());assert len(prompt['prompt_token_ids'])==131072
assert hashlib.sha256((a.case/'prompt.json').read_bytes()).hexdigest()==result['prompt_sha256']
report=dict(case=a.case.name,status=result['status'],context=131072,init_s=result.get('init_s'),
            policy=result['policy'],config=result['config'],runs=[],traces=[],model_quality_qualified=False,
            result_sha256=hashlib.sha256((a.case/'result.json').read_bytes()).hexdigest())
for run in result['runs']:
 if run['status']!='COMPLETED':continue
 ranks=run['ranks'];assert len(ranks)==8 and sorted(r['rank']for r in ranks)==list(range(8))
 for r in ranks:
  assert len(r['windows'])==256 and [w['index']for w in r['windows']]==list(range(1,257))
  assert all(w['completed'] and w['query_length']==512 for w in r['windows'])
 assert len(run['output_ids'])==1 and run['repeat_token_match']
 groups=[];steps=run['steps'];aligned=len(steps)==256 and sum(s['new_tokens']for s in steps)==1
 if aligned:
  for begin,end in ((0,16),(16,64),(64,128),(128,192),(192,256)):
   values=[s['ms']for s in steps[begin:end]]
   groups.append(dict(tokens_begin=begin*512,tokens_end=end*512,chunks=len(values),sum_ms=sum(values),median_chunk_ms=statistics.median(values),max_chunk_ms=max(values)))
 report['runs'].append(dict(label=run['label'],wall_s=run['wall_s'],prompt_completion_tokens_s=131072/run['wall_s'],
                           contains_profile_overhead=run['contains_profile_overhead'],coordinator_steps=len(steps),
                           step_to_chunk_alignment_proven=aligned,coordinator_regions=groups,
                           worker_host_call_median_ms_by_rank={r['rank']:statistics.median(w['host_call_ns']/1e6 for w in r['windows'])for r in ranks},
                           memory=[{k:v for k,v in r.items()if k!='windows'}for r in ranks]))
 for rank in ranks:
  for w in rank['windows']:
   if not w['profiled']:continue
   assert w['profile_start_code']==0 and w['profile_stop_code']==0 and w['trace_entries']>1
   path=a.case/'prefill-workers'/Path(w['trace']).name
   summary,rows=read(path,require_gate=False);assert summary['valid'],summary['errors']
   summary['scope']='one marked prefill chunk, first-to-last TPC/MME; profiled repeat only, includes engine stalls'
   ops={k.removeprefix('op/'):v for k,v in summary['union_ms'].items()if k.startswith('op/')}
   top=sorted(ops.items(),key=lambda x:x[1],reverse=True)[:25]
   report['traces'].append(dict(rank=rank['rank'],chunk=w['index'],prefix_tokens=(w['index']-1)*512,
                               query_length=w['query_length'],kv_blocks=w['kv_blocks'],summary=summary,
                               top_operator_union_ms=top,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
report['expected_six_profiles_complete']=sorted((r['rank'],r['chunk'])for r in report['traces'])==[(r,c)for r in (0,3)for c in (1,128,256)]
report['limits']=['Prompt completion includes admission, scheduler and first-token sampling; model initialization reported separately.',
                 'Profiled repeat includes synchronization/profiler overhead and is not a clean warm-throughput benchmark.',
                 'Unprofiled worker host-call durations do not independently prove device completion.',
                 'Engine/operator unions overlap; gaps do not prove removable CPU or communication cost.',
                 'No physical HBM byte counter or FLOP issue counter was collected.']
a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],runs=[{k:r[k]for k in ('label','wall_s','prompt_completion_tokens_s','coordinator_steps')}for r in report['runs']],profiles=len(report['traces']))))
