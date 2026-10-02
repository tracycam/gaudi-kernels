"""Use the slowest rank per paired trial, never the fastest/average rank."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from fixtures import metric

def aggregate(rows):
    groups={}
    for r in rows:
        if r['stage']!='timing':continue
        assert r['timing_valid'] and r['collectives']==69*r['repeats']
        key=(r['phase'],r['trial']);group=groups.setdefault(key,[]);group.append(r)
    result=[]
    for (phase,trial),group in sorted(groups.items()):
        assert {r['rank'] for r in group}==set(range(8)) and len(group)==8,('missing/duplicate rank',phase,trial)
        assert len({r['repeats'] for r in group})==len({r['mode'] for r in group})==1
        n=group[0]['repeats'];result.append(dict(phase=phase,trial=trial,mode=group[0]['mode'],slowest_rank_wall_us_per_69=max(r['wall_us'] for r in group)/n,slowest_rank_event_us_per_69=max(r['event_us'] for r in group)/n,latest_submit_us_per_69=max(r.get('submit_us',0) for r in group)/n))
    assert {r['phase'] for r in result}==set(range(4))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path);p.add_argument('--fixtures',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        rows=[dict(stage='timing',rank=rank,phase=phase,trial=0,mode='FP32_AG_sum' if phase in (1,2) else 'FP32_AR',timing_valid=True,collectives=69,repeats=1,wall_us=100+rank,event_us=99+rank) for phase in range(4) for rank in range(8)]
        r=aggregate(rows);assert all(v['slowest_rank_wall_us_per_69']==107 for v in r)
        try:aggregate(rows[:-1])
        except AssertionError:pass
        else:raise AssertionError('missing rank accepted')
        print(json.dumps(dict(all_pass=True,slowest_rank_not_average=True,missing_rank_rejected=True)));return
    assert a.case and a.fixtures
    status=json.loads((a.case/'exit.json').read_text());assert status['runner_exit_code']==0
    handoff=status.get('comparison','ag_sum')=='ar_handoff'
    paired_consumers=status.get('comparison')=='ar_ag_consumers'
    inplace_comparison=status.get('comparison')=='ag_inplace'
    two_stream=status.get('comparison')=='ag_two_stream'
    packet_only=status.get('packet_only',False)
    stages=1 if packet_only else 69
    results=[]
    for kind in ['exact','ordinary']:
        rows=[];d=a.case/kind;fixture=a.fixtures/kind;ref=np.fromfile(fixture/'oracle-fp64.bin',np.float64).reshape(69,6144);ab=np.fromfile(fixture/'absolute-fp64.bin',np.float64).reshape(69,6144);ordered=np.fromfile(fixture/'ordered-fp32.bin',np.float32).reshape(69,6144)
        for rank in range(8):
            records=[json.loads(s) for s in (d/f'rank{rank}.jsonl').read_text().splitlines()];assert len([r for r in records if r['stage']=='quality'])==8
            for r in records:
                if r['stage']=='quality':assert r['bad']==0
                if r['stage']=='timing' and status.get('prequeue'):assert r['prequeued'] and r['gate_query']==22 and r['gate_us']>0
            rows.extend(records)
            for phase in range(4):
                alternative=phase in (1,2);gather=inplace_comparison or two_stream or (alternative and not handoff);tag=(('agdual' if alternative else 'agserial') if two_stream else ('agin' if alternative else 'agout') if inplace_comparison else ('agconsumer' if alternative else 'arconsumer') if paired_consumers else (('handoff' if handoff else 'ags') if alternative else 'ar'))+str(phase)
                api=[json.loads(s) for s in (d/f'rank{rank}-{tag}-api.jsonl').read_text().splitlines()];comm=[r for r in api if r['api'].startswith('hccl')]
                assert len(comm)==stages and all(r['count']==6144 and r['dtype']==7 and r['status']==0 and r['api']==('hcclAllGather' if gather else 'hcclAllReduce') for r in comm)
                if two_stream:
                    from stream_audit import check
                    check(api,stages,alternative)
                if inplace_comparison:
                    assert all(r['send_address']==r['recv_address']+(rank if alternative else 8)*24576 for r in comm)
                launches=[r for r in api if r['api'].startswith('synLaunch')]
                assert len(launches)==(2*stages if inplace_comparison or two_stream or paired_consumers or (handoff and alternative) else stages if gather else 0)
                assert all(r['status']==0 for r in launches)
                for suffix in ['', '-post']:
                    got=np.fromfile(d/f'rank{rank}-{tag}{suffix}.bin',np.float32).reshape(stages,6144);m=metric(got,ref[:stages],ab[:stages]);assert m['bad']==0
                    if gather:assert np.array_equal(got.view(np.uint32),ordered[:stages].view(np.uint32))
                    if kind=='exact':assert np.array_equal(got.astype(np.float64),ref[:stages])
                    if inplace_comparison or two_stream:
                        raw=np.fromfile(d/f'rank{rank}-{tag}{suffix}-gathered.bin',np.uint32).reshape(8,6144)
                        all_inputs=np.fromfile(fixture/'all-inputs.bin',np.uint32).reshape(8,69,6144)
                        assert np.array_equal(raw,all_inputs[:,stages-1,:])
        trials=[] if packet_only else aggregate(rows);phases=[]
        for phase in ([] if packet_only else range(4)):
            vals=[r for r in trials if r['phase']==phase];phases.append(dict(phase=phase,mode=vals[0]['mode'],median_slowest_rank_wall_us_per_69=statistics.median(r['slowest_rank_wall_us_per_69'] for r in vals),median_slowest_rank_event_us_per_69=statistics.median(r['slowest_rank_event_us_per_69'] for r in vals)))
        results.append(dict(fixture=kind,trials=trials,phases=phases,quality=[r for r in rows if r['stage']=='quality']))
    report=dict(all_pass=True,records=results,engine=status['engine'],packet_only=packet_only,prequeue=status.get('prequeue',False),comparison=status.get('comparison','ag_sum'),scope='untimed one-stage packet diagnostic' if packet_only else '69 24KiB FP32 collectives; optional dependent TPC producer/consumer; no whole-model critical-path/TPS inference',physical_HBM_or_NIC_counters_measured=False)
    (a.case/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS')
if __name__=='__main__':main()
