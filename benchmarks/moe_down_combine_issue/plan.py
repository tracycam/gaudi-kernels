"""Offline read/issue budget for a same-resident-byte N256 down+combine owner."""
import argparse,collections,csv,hashlib,importlib.util,json,re,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
archive=a.canonical/'artifacts/builds/moe-production-issue/final-a'
source=archive/'offline/offline-a/down.s';text=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='af1863e523b6fef932ef2d42dff9442331ceeb2d7b33cd5fa7845b2af03bacb9'
body=text.split('.LBB0_3:\n')[1].split('.LBB0_6:')[0]
rows=[[s.strip() for s in row.split(';')] for row in body.splitlines()];assert len(rows)==289
pruned=[]
for i,row in enumerate(rows):
    row=list(row)
    for slot,op in enumerate(row):
        drop=(bool(re.search(r'\b(?:D8|D10|V8|V9|V10|V11)\b',op))
              or bool(re.search(r'lookup_2c.*\b(?:D20|D26|D32)\b',op))
              or op.startswith('mac.f32 V4,') or op.startswith('mac.f32 V5,')
              or op.startswith('mac.f32 V6,') or op.startswith('mac.f32 V7,'))
        # V36/37 remain activation buffers earlier in the body: prune only tail.
        if i>240 and (op.startswith('ld_tnsr') and 'V36,' in op or
                      op.startswith('convert.u8') and 'D36,' in op or
                      op.startswith('shl.i16 V36,') or op.startswith('shl.i16 V37,') or
                      op.startswith('convert.bf16') and ('D32,' in op or 'D34,' in op)):
            drop=True
        if drop:row[slot]='nop'
    pruned.append(row)
(a.output/'n256-unscheduled-body.sfrag').write_text('\n'.join('; '.join(r) for r in pruned)+'\n')
def counts(rows):
    return dict(packets=len(rows),active_slots=[sum(r[i]!='nop' for r in rows) for i in range(4)],
                operations=[dict(collections.Counter(r[i].split()[0] for r in rows if r[i]!='nop')) for i in range(4)],
                full_nop=sum(r==['nop']*4 for r in rows))
# Exhaustive address/byte ownership on a production-size shard, no weight expansion on device.
spec=importlib.util.spec_from_file_location('legacy',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
N,K=6144,256
n=np.arange(N,dtype=np.uint32)[:,None];k=np.arange(K//2,dtype=np.uint32)[None,:]
raw=((n*37+k*11+(n//128)*3)%256).astype(np.uint8)
scale=((np.arange(N,dtype=np.uint32)[:,None]*5+np.arange(K//32)[None,:]*7)%255).astype(np.uint8)
packed,sc=m.pack(raw,scale);w128=packed.reshape(-1,128);s256=sc.reshape(-1,256)
assert np.shares_memory(packed,w128) and np.shares_memory(sc,s256)
seen_w=np.zeros(packed.size,dtype=np.uint8);seen_s=np.zeros(sc.size,dtype=np.uint8)
for tile in range(N//256):
    nb,half=divmod(tile,2)
    for z in range(K):
        row=nb*K*2+half+z*2;v=w128[row];wanted=((raw[tile*256:tile*256+256,z//2]>>(4*(z%2)))&15)
        decoded=np.concatenate((v&15,v>>4));assert np.array_equal(decoded,wanted)
        seen_w[row*128:(row+1)*128]+=1
    for group in range(K//32):
        row=nb*(K//32)*2+half+group*2;assert np.array_equal(s256[row],scale[tile*256:tile*256+256,group])
        seen_s[row*256:(row+1)*256]+=1
assert np.all(seen_w==1) and np.all(seen_s==1)
# Historical accumulator layout, matching precision_fix.combine's even/odd restore.
logical=np.arange(N);internal=logical.reshape(-1,128)[:,np.r_[np.arange(0,128,2),np.arange(1,128,2)]]
restored=np.empty_like(internal);restored[:,0::2]=internal[:,:64];restored[:,1::2]=internal[:,64:]
assert np.array_equal(restored.ravel(),logical)
np.savez(a.output/'lane-and-byte-fixture.npz',checkpoint_packed=raw,checkpoint_scales=scale,resident_packed=packed,resident_scales=sc,internal_lanes=internal,restored_lanes=restored)
# Actual old-arm down/combiner span from the already sealed paired profile.
csvfile=archive/'remote/module7-a611991/profiles/gp-scale-tail-loadsafe-profile-a/trace/gp-scale-tail-loadsafe-profile-a_accel5_0001-1000_analyzed_nodes.csv'
allrows=list(csv.DictReader(csvfile.open()));gp=sorted([r for r in allrows if r['Op Type']=='gk_moe_gp_folded_v1'],key=lambda r:float(r['Start time of node']))[-96:]
graph=gp[0]['Graph Name'];chosen={r['Start time of node'] for r in gp};ordered=sorted([r for r in allrows if r['Graph Name']==graph],key=lambda r:float(r['Start time of node']));timing=[]
for i in range(0,len(ordered),12):
    chunk=ordered[i:i+12];g=next(r for r in chunk if r['Op Type']=='gk_moe_gp_folded_v1')
    if g['Start time of node'] not in chosen:continue
    d=next(r for r in chunk if r['Op Type']=='downact_direct_down_broadcast');c=next(r for r in chunk if r['Op Type']=='pf_combine_f32')
    timing.append(dict(down_us=float(d['Duration (us)']),combine_us=float(c['Duration (us)']),span_us=float(c['End time of node'])-float(d['Start time of node']),gap_us=float(c['Start time of node'])-float(d['End time of node'])))
assert len(timing)==96
result=dict(device_used=False,status='OFFLINE_LAYOUT_AND_BUDGET_ONLY',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    baseline=counts(rows),pruned_unscheduled=counts(pruned),
    lane_fixture=dict(N=N,K=K,checkpoint_nibble_order='K-even low',resident_layout='historical unpermuted N512',all_codes_and_scales_equal=True,all_resident_bytes_covered_exactly_once=True,no_repacking=True,CPU_views_share_storage=True,weight_bytes=packed.size,scale_bytes=sc.size,device_descriptor_alias_validated=False),
    M1_top8_budget=dict(old_tasks=96,fused_tasks=24,old_route_tiles=96,new_route_tiles=192,
        old_K32_body_packets_per_nominal_core=4*8*289,new_uncompressed_K32_body_packets_per_core=8*8*289,
        new_active_VPU_issue_floor_per_core=64*counts(pruned)['active_slots'][2],old_active_VPU_issue_per_core=32*counts(rows)['active_slots'][2],
        packed_read_valid_bytes=8*N*K//2,scale_read_valid_bytes=8*N*K//32,total_weight_valid_bytes=8*N*K*17//32,
        removable_partial_write_read_bytes=2*8*N*4,output_write_bytes=N*4,
        old_activation_tensor_loads=96*8,new_activation_tensor_loads=24*8*8,
        physical_HBM_claim=False),
    recurrence_sensitivity=[dict(assumed_BF16_MAC_latency_packets=L,lower_bound_body_packets=max(counts(pruned)['active_slots'][2],31*L+2),scope='issue/dependence lower bound; excludes load pipeline, scale tail, prologue and routing') for L in (2,4,6,8)],
    actual_control_profile=dict(samples=96,medians={k:statistics.median(r[k] for r in timing) for k in timing[0]},source_csv_sha256=hashlib.sha256(csvfile.read_bytes()).hexdigest(),scope='observed down/combine subspan in original full chain; no fused performance measurement'),
    interface=dict(proposed='gaudi_down_combine_m1::fused(w128,s256,gate_bf16,lut,ids_i32,routing_f32,directions)->FP32[1,6144]',
        fixed_contract='M1,top8,N6144,K256,valid device expert IDs; original finite fast arithmetic domain',
        graph_inputs='persistent zero-copy views of original resident weight/scales, BF16[8,256], LUT, ids[1,8], FP32 routing[1,8], original directions',
        graph_outputs='only natural-order FP32[1,6144]; no route-partial tensor',
        pending='actual assembly wrapper/scheduler, simulator gate, framework alias/placement, device bits and full-chain timing'))
(a.output/'plan.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','baseline','pruned_unscheduled','M1_top8_budget','actual_control_profile')}))
