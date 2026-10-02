"""Attribute the bounded M1 down profile by actual recipes and complete replay order."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--summary-out',type=Path);a=p.parse_args()
torch.set_num_threads(2)
archive=a.case.parents[1];manifest=json.loads((archive/'remote-sha256.json').read_text())
def checked(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][str(path.relative_to(archive))]['sha256']
    return path
gate=json.loads((a.case/'result.json').read_text())
checked(a.case/'result.json')
assert gate['status']=='PASS_FULL_CHAIN_BITS_AND_ABBA' and gate['rows']==[1]
assert len(gate['timing'])==12 and len(gate['records'])==1
assert len(gate['records'][0]['checks'])==12
references={};words=0
for state in ('initial','hot_changed','cold_changed','zero_routes'):
    data=torch.load(checked(a.case/'m1'/(state+'.pt')),weights_only=False)
    ref=data['variants']['deployed289'][0];references[state]=ref
    for samples in data['variants'].values():
        assert len(samples)==1
        for x,y in zip(ref,samples[0]):
            assert x.shape==y.shape==(1,6144) and x.dtype==y.dtype==torch.float32 and y.isfinite().all()
            assert torch.equal(x.view(torch.uint8),y.view(torch.uint8));words+=y.numel()
for row in gate['timing']:
    values=torch.load(checked(a.case/row['output_file']),weights_only=False)
    for x,y in zip(references[row['state']],values):
        assert torch.equal(x.view(torch.uint8),y.view(torch.uint8));words+=y.numel()
mapping={}
for row in gate['records'][0]['graphs']:
    assert len(row['postgraph_names'])==1
    mapping[row['postgraph_names'][0]]=row['variant']
traces=list((a.case/'trace').glob('*.json'));assert len(traces)==1
report=analyze(traces[0],a.case/'post_graph.json',list(mapping))
ordered=sorted(report['invocations'],key=lambda r:r['start_us'])
expected=[]
for arm in gate['timing']:
    assert arm['warmup']==3 and arm['replays']==4
    expected += [dict(arm,phase='warmup' if i<3 else 'timed')for i in range(7)]
assert len(expected)==84
prefix,samples=ordered[:-len(expected)],ordered[-len(expected):]
assert Counter(r['recipe']for r in prefix)==Counter({name:5 for name in mapping})
for row,arm in zip(samples,expected):
    assert mapping[row['recipe']]==arm['variant']
    row.update(variant=arm['variant'],state=arm['state'],phase=arm['phase'],arm_index=arm['arm_index'])
summary=[]
for state in ('initial','hot_changed','cold_changed'):
    for variant in ('deployed289','candidate242'):
        selected=[r for r in samples if (r['state'],r['variant'],r['phase'])==(state,variant,'timed')]
        assert len(selected)==8
        nodes=defaultdict(list);cores=defaultdict(set);gaps=[]
        for row in selected:
            seen=Counter();intervals=[]
            for node in row['nodes']:
                seen[node['op']]+=1;key=node['op']+'#'+str(seen[node['op']])
                nodes[key].append(node['envelope_us']);cores[key].add(node['engine_count'])
                intervals.append((node['start_offset_us'],node['end_offset_us']))
            covered=0;end=0
            for begin,stop in sorted(intervals):
                covered+=max(0,stop-max(begin,end));end=max(end,stop)
            gaps.append(row['envelope_us']-covered)
        summary.append(dict(state=state,variant=variant,invocations=len(selected),
            recipe_envelope_median_us=statistics.median(r['envelope_us']for r in selected),
            uncovered_between_node_envelopes_median_us=statistics.median(gaps),
            node_envelope_median_us={k:statistics.median(v)for k,v in nodes.items()},
            engine_counts={k:sorted(v)for k,v in cores.items()}))
report.update(summary=summary,ordered_replay_samples=samples,
    root_checked_output_words=words,
    sequence='15 qualification invocations (5 per recipe), then 84 warmup/timed invocations; all recipe identities and complete physical nodes checked.',
    scope='Profiler-perturbed whole-chain device trace. Node envelopes include memory/issue stalls; not useful arithmetic cycles or an end-to-end TPS gain.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
if a.summary_out:
    compact={k:v for k,v in report.items()if k not in ('invocations','ordered_replay_samples')}
    compact.update(full_analysis=str(a.out),full_analysis_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())
    a.summary_out.parent.mkdir(parents=True,exist_ok=True);a.summary_out.write_text(json.dumps(compact,indent=2)+'\n')
print(json.dumps(summary))
