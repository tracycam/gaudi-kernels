#!/usr/bin/env python3
"""Bounded CPU route/reference gates. This does not execute MXFP4 MACs."""
import argparse,copy,hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from planner import Capacity,assign,fixture,validate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_moe_plan import MoePlan


def ordered_combine(values,weights):
    # Fixture values/weights are dyadic on 2^-4 lattice with magnitudes <=2^26;
    # product+sum is exactly representable in binary64 (<2^53 scaled integer).
    # One RN32 therefore models each FMA exactly for THESE bounded fixtures.
    acc=np.zeros((values.shape[0],values.shape[2]),np.float32)
    for slot in range(values.shape[1]):
        acc=(values[:,slot].astype(np.float64)*weights[:,slot,None].astype(np.float64)+acc.astype(np.float64)).astype(np.float32)
    return acc


def numeric_fixture(plan):
    t,r=plan.capacity.tokens,plan.capacity.routes;length=t*r
    values=((np.arange(length*4).reshape(t,r,4)%97)-48).astype(np.float32)
    weights=((np.arange(length).reshape(t,r)%7)-3).astype(np.float32)/4
    if r>=4:
        values[:,:4,0]=np.array([2**24,1,-2**24,1],np.float32);values[:,4:,0]=0;weights[:,:4]=1
    packed=np.zeros((len(plan.row_to_route),4),np.float32)
    for row,route in enumerate(plan.row_to_route):
        if route>=0:packed[row]=values.reshape(-1,4)[route]
    restored=packed[plan.inverse].reshape(t,r,4)
    assert np.array_equal(values.view(np.uint32),restored.view(np.uint32))
    expected=ordered_combine(values,weights);actual=ordered_combine(restored,weights)
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    if r>=4:assert not np.array_equal(ordered_combine(restored[:,::-1],weights[:,::-1]).view(np.uint32),expected.view(np.uint32))
    return expected


def bound_gate():
    checked=0
    # Independent dynamic program maximizes sum ceil(count/C), every count<=T.
    # This does not reuse the closed-form expression under test.
    for e in range(1,9):
        for t in range(1,10):
            for request in (1,2,3,4,8,16):
                c=min(t,request);dp={0:0}
                for _ in range(e):
                    nxt={}
                    for used,tiles in dp.items():
                        for count in range(t+1):
                            cost=(count+c-1)//c
                            nxt[used+count]=max(nxt.get(used+count,-1),tiles+cost)
                    dp=nxt
                for r in range(1,e+1):assert dp[t*r]==Capacity(t,r,e,request).max_tiles;checked+=1
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    records=[];numerical_words=0
    for t in (1,2,8,32,128,512,513):
        for r in (1,2,8):
            for request in (16,32):
                cap=Capacity(t,r,384,request)
                for kind in ('uniform','hot','skew'):
                    ids=fixture(t,r,384,kind);plan=assign(ids,cap);output=numeric_fixture(plan);numerical_words+=output.size
                    cost=cap.costs(sum(v>0 for v in plan.tile_valid_rows),sum(v>0 for v in plan.counts))
                    old={mode:MoePlan(mode=mode).costs(t,r,384) for mode in ('capacity_t','bucket')} if t>1 else {'capacity_t':{'row_slots':384,'mme_nodes':480,'scope':'math-only M1 bound; public grouped API rejects T1'}}
                    records.append(dict(distribution=kind,requested_tile_rows=request,route_sha256=hashlib.sha256(np.array(ids,np.int32).tobytes()).hexdigest(),inverse_sha256=hashlib.sha256(np.array(plan.inverse,np.int32).tobytes()).hexdigest(),reference_sha256=hashlib.sha256(output.tobytes()).hexdigest(),**cost,existing_plans=old))
                    if (t,r,request,kind)==(513,8,16,'skew'):
                        a.output.with_name('route-witness.json').write_text(json.dumps({'ids':ids,'plan':asdict(plan)},indent=2)+'\n')
    dp=bound_gate();negative=[]
    ids=fixture(32,8,384,'skew');original=assign(ids,Capacity(32,8))
    for name in ('duplicate_route','drop_route','wrong_expert','wrong_inverse'):
        bad=copy.deepcopy(original)
        if name=='duplicate_route':bad.row_to_route[1]=bad.row_to_route[0]
        if name=='drop_route':bad.row_to_route[0]=-1
        if name=='wrong_expert':bad.tile_expert[0]=(bad.tile_expert[0]+1)%384
        if name=='wrong_inverse':bad.inverse[0]=bad.inverse[1]
        try:validate(ids,bad)
        except AssertionError:negative.append(name)
        else:raise AssertionError('fault accepted: '+name)
    for ids in ([[1,1]],[[0,384]]):
        try:assign(ids,Capacity(1,2))
        except ValueError:negative.append('invalid_or_duplicate_ID')
        else:raise AssertionError('bad IDs accepted')
    report={'status':'CPU_SPECIFICATION_PASS','device_implemented':False,'cases':len(records),'numeric_output_words':numerical_words,'independent_capacity_DP_cases':dp,'fault_injections_rejected':negative,'ordered_combine':'Original slots ascending; bounded dyadic FMA-order fixture, not MXFP4 full-MAC qualification. Reversed order rejected for every R8 case.','records':records}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
