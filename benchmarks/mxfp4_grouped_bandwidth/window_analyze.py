"""Audit decoder-only changes against the retained finite-FP32-MAC baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import numpy as np
from analyze import graph_audit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sram_ports'))
# 'analyze' is already the grouped module. Load endpoint trace under a new name.
import importlib.util
spec=importlib.util.spec_from_file_location('port_analysis',Path(__file__).resolve().parents[1]/'sram_ports/analyze.py')
port=importlib.util.module_from_spec(spec);spec.loader.exec_module(port)

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.root;records=[];digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for case in sorted((root/'results').iterdir()):
    status=json.loads((case/'exit.json').read_text());assert status['runner_exit_code']==0 and status['module']==5 and status['source_identity_verified']
    rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{"stage":')]
    plan=next(x for x in rows if x['stage']=='plan');gate=next(x for x in rows if x['stage']=='correctness')
    assert gate['bad']==0 and gate['max_componentwise_backward']<2e-6
    graph=graph_audit(case,plan);assert graph['all_expanded_weights_sram']
    baseline=root/'results'/f'screen-e{plan["experts"]}-0-base'
    if case.name.startswith('geometry-'):
        baseline=root/'results'/('-'.join(case.name.split('-')[:-1])+'-base')
    hashes={}
    for b in range(plan['banks']):
        name=f'output_bank{b}.bin';assert (case/name).read_bytes()==(baseline/name).read_bytes(),(case,name)
        actual=np.fromfile(case/name,dtype='<f4');ref=np.fromfile(case/f'oracle_bank{b}.bin',dtype='<f8')
        assert np.isfinite(actual).all() and actual.size==plan['experts']*plan['N']*plan['M_cap']
        assert np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)<2e-6
        hashes[name]=digest(case/name)
    for name in ['packed.bin','scales.bin','activation.bin','expert_ids.bin']:
        assert digest(case/name)==digest(baseline/name),(case,name)
    us=statistics.median(x['event_us'] for x in rows if x['stage']=='timing')
    count=plan['experts']*plan['N']*plan['K']
    r=dict(case=case.name,plan=plan,gate=gate,graph=graph,event_median_us=us,
           bitwise_identical_to_baseline=True,output_sha256=hashes,
           effective_payload_TBps=count*.5/us/1e6,effective_original_TBps=count*17/32/us/1e6,
           expanded_bytes_per_complete_time_TBps=count*2/us/1e6)
    traces=list((case/'trace').glob('*_7.json'))
    if traces:r['profile']=port.trace(traces[0],None)
    records.append(r)
by={r['case']:r for r in records};abba=[]
for e in (8,32):
    names=[f'abba-e{e}-{i}-{name}' for i,name in enumerate(('base','k8','k8','base'))]
    arms=[by[n] for n in names];assert all('profile' not in x for x in arms)
    before=statistics.mean(arms[i]['event_median_us'] for i in (0,3));after=statistics.mean(arms[i]['event_median_us'] for i in (1,2))
    count=e*512*6144
    abba.append(dict(experts=e,M=2,N=512,K=6144,arms=names,baseline_us=before,candidate_us=after,
                     latency_reduction=1-after/before,original_TBps=count*17/32/after/1e6,
                     payload_TBps=count*.5/after/1e6,expanded_feed_over_complete_time_TBps=count*2/after/1e6,
                     target_3p2_TBps_achieved=count*2/after/1e6>=3.2))
isa=[]
for name in ('base','k4','k8','k16'):
    d=root/name;src=(d/'decode.s').read_text();code=src[src.index('main:'):src.index('$func_end0:')]
    dis=(d/'decode.o.dis').read_text()
    local=[l for l in dis.splitlines() if re.search(r'\b(?:ld|st)_l(?:_v)?\b',l) and 'mmio' not in l]
    regs=[int(v)+(k=='D') for k,v in re.findall(r'\b(V|D)(\d+)\b',dis)]
    isa.append(dict(build=name,elf_sha256=digest(d/'decode.o'),highest_vector_register=max(regs),
                    static_local_memory_ops=local,static_branch_count=len(re.findall(r'\bjmpr\b',code)),
                    static_hardware_loop_count=len(re.findall(r'\bloop ',code)),
                    scope='Static code only; not dynamic instruction/cycle counts'))
    assert not local
out=dict(scope='Single-module W4A16 grouped projection, not full MoE/model throughput; finite-MAC baseline permits canonical zero',
         source_identity=json.loads((root/'source-identity.json').read_text()),ABBA=abba,records=records,isa=isa,
         no_weight_read_amplification_scope='Original packed+scale ownership, disjoint compiler expert slices and unchanged per-element source loads verified; physical HBM transactions not independently measured')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(ABBA=abba,isa=isa),indent=2))
