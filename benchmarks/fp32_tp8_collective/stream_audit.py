"""Validate the complete public API dependency chain, including scratch reuse."""
from copy import deepcopy

def check(rows, stages, dual):
    assert rows and rows[-1]['api']=='synStreamSynchronize'
    assert all(r['status']==0 for r in rows)
    terminal=rows[-1];steps=rows[:-1];size=7 if dual else 3
    assert len(steps)==stages*size
    compute=None;communication=None;events=set()
    for stage in range(stages):
        group=steps[stage*size:(stage+1)*size]
        expected=['synLaunch','synEventRecord','synStreamWaitEvent','hcclAllGather','synEventRecord','synStreamWaitEvent','synLaunch'] if dual else ['synLaunch','hcclAllGather','synLaunch']
        assert [r['api']for r in group]==expected
        c=group[0]['stream'];n=group[3 if dual else 1]['stream']
        assert c and n and (compute is None or compute==c)
        assert communication is None or communication==n
        compute,communication=c,n
        assert group[-1]['stream']==c and terminal['stream']==c
        collective=group[3 if dual else 1]
        assert collective['count']==6144 and collective['dtype']==7
        assert collective['send_address']==collective['recv_address']+8*24576
        if dual:
            assert c!=n
            p,w,a,v=group[1],group[2],group[4],group[5]
            assert p['stream']==c and w['stream']==n and a['stream']==n and v['stream']==c
            assert p['event']==w['event'] and a['event']==v['event']
            assert p['event'] and a['event'] and p['event']!=a['event']
            assert p['event']not in events and a['event']not in events
            events.update((p['event'],a['event']))
            assert w['count']==v['count']==0
        else:assert c==n
    return dict(stages=stages,compute_stream=compute,collective_stream=communication,
                unique_dependency_events=len(events),host_syncs=1,
                scope='Public call order and stream/event handles; not physical command latency')

def self_test():
    rows=[]
    for i in range(3):
        for api,stream,event in [('synLaunch',1,0),('synEventRecord',1,10+2*i),('synStreamWaitEvent',2,10+2*i),('hcclAllGather',2,0),('synEventRecord',2,11+2*i),('synStreamWaitEvent',1,11+2*i),('synLaunch',1,0)]:
            rows.append(dict(api=api,stream=stream,event=event,count=6144 if api=='hcclAllGather' else 0,dtype=7,status=0,send_address=8192+8*24576,recv_address=8192))
    rows.append(dict(api='synStreamSynchronize',stream=1,status=0));check(rows,3,True)
    cases=[]
    x=deepcopy(rows);x[2]['event']=99;cases.append(x)
    x=deepcopy(rows);x[5]['stream']=2;cases.append(x)
    x=deepcopy(rows);x[8]['event']=x[9]['event']=10;cases.append(x)
    x=deepcopy(rows);x.insert(7,deepcopy(rows[-1]));cases.append(x)
    x=deepcopy(rows);x[3]['send_address']-=24576;cases.append(x)
    for x in cases:
        try:check(x,3,True)
        except AssertionError:pass
        else:raise AssertionError('Broken handoff accepted')
    serial=[]
    for i in range(3):
        group=deepcopy(rows[i*7:(i+1)*7])
        for r in (group[0],group[3],group[6]):r['stream']=1;serial.append(r)
    serial.append(rows[-1]);check(serial,3,False)
    return dict(valid_chains=2,rejected_mutations=len(cases))

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
