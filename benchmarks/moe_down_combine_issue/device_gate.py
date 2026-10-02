"""Same-owner M1 public graph: GP/gate + deployed down/combine versus fused ISA.

Only tools/run_device_probe.py may launch this file. P4/P6 are experiments, not
production policy choices. Numerical failure prevents all timing in this case.
"""
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--variants',nargs='+',choices=('p6','p4'),default=['p6','p4']);p.add_argument('--streaming',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--timing',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')=='7' and os.environ.get('HABANA_VISIBLE_MODULES')=='7'
assert os.environ.get('PT_HPU_LAZY_MODE')=='1' and len(a.variants)==len(set(a.variants))
manifest=json.loads(a.manifest.read_text())
for name,item in manifest['files'].items():
    path=(root/item['path']).resolve();assert path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],name
assert not manifest['offline_Meta_stub']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
os.environ.update(PT_ENABLE_INT64_SUPPORT='0',ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),UNIFIED_PRECISION_ROUTER='1',GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0')
(out/'graphs').mkdir()
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(2)
fixture=root/manifest['executor_fixture'];snapshot=json.loads((fixture/'snapshot.json').read_text())
for name,sha in snapshot['files_sha256'].items():assert hashlib.sha256((fixture/name).read_bytes()).hexdigest()==sha,name
sys.path.insert(0,str(fixture/'executor'));import batch_ops,precision_ops
for key in ('gp_torch','down_torch','candidate_torch'):torch.ops.load_library(str(root/manifest['files'][key]['path']))
from kernels.packing import pack,unpack
ops=torch.ops.gaudi_down_combine_isa

def sync():hc.mark_step();torch.hpu.synchronize()
def same(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def summary(x,y):
    u=x.contiguous().view(torch.int32).flatten();v=y.contiguous().view(torch.int32).flatten();bad=(u!=v).nonzero().flatten()
    return dict(pairs=u.numel(),bit_mismatches=bad.numel(),old_finite=bool(torch.isfinite(x).all()),new_finite=bool(torch.isfinite(y).all()),first_bad=[dict(index=int(i),old_bits=hex(int(u[i])&0xffffffff),new_bits=hex(int(v[i])&0xffffffff)) for i in bad[:8]])
def graph_ids():
    path=out/'post_graph.json';files=[path] if path.is_file() else sorted(path.rglob('*.json')) if path.is_dir() else []
    return {g['id'] for f in files for g in json.loads(f.read_text())['graphs']}
result=dict(status='STARTED',bridge_integer_mode='PT_ENABLE_INT64_SUPPORT=0; logical Long is physical I32, enforced by glue',device_used=True,records=[],direct=[],timing=[],model_quality_qualified=False,physical_HBM_measured=False,scope='Fixed M1/top8 original-owner public graph fusion; no private acquisition/replay APIs',event_scope='Capture-stream event includes host replay gaps; synchronized wall separate')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
    torch.manual_seed(928417);rng=np.random.default_rng(928417);E=384 if a.streaming else 24
    weights=[]
    for n,k in ((512,6144),(6144,256)):
        wp=[];sp=[]
        for e in range(E):
            packed,scales=pack(rng.integers(0,256,(n,k//2),dtype=np.uint8),rng.integers(115,124,(n,k//32),dtype=np.uint8))
            wp.append(packed.reshape(-1,256));sp.append(scales.reshape(-1,512))
        weights.extend([torch.from_numpy(np.concatenate(wp)),torch.from_numpy(np.concatenate(sp))]);del wp,sp
    torch.save(dict(zip(('gp','gs','dp','ds'),weights)),out/'prepared-weights.pt')
    result['weights']=dict(experts=E,layout='HistoricalN512=1',format='original E2M1/E8M0',bytes=sum(t.numel()*t.element_size() for t in weights),down_bytes=weights[2].numel()+weights[3].numel(),source='bounded synthetic fixture',sha256=hashlib.sha256((out/'prepared-weights.pt').read_bytes()).hexdigest(),minimum_scale=int(min(weights[1].min(),weights[3].min())),maximum_scale=int(max(weights[1].max(),weights[3].max())))
    gp,gs,dp,ds=[t.to('hpu') for t in weights]
    q=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.bfloat16)
    lut=torch.stack((q[torch.arange(256)%16],q[torch.arange(256)//16]),dim=1).reshape(1,512).to('hpu')
    dirs=torch.empty((2,256),dtype=torch.uint8)
    for parity in range(2):
        for byte in range(256):
            lane=byte//4;dirs[parity,byte]=(lane%16)//2+((lane//16)%2)*32+(128 if lane%2==parity else 0)
    directions=dirs.to('hpu');sync()
    def fused(variant,gate,ids,routing):
        # Public logical reshapes preserve producer edges; actual alias placement
        # remains a mandatory postgraph gate, never inferred from tensor shape.
        w128=ops.weight_view(dp,False);s256=ops.weight_view(ds,True)
        return getattr(ops,variant)(w128,s256,gate,lut,ids,routing,directions,1,True)
    def down_reference(gate,ids,routing):
        answer=torch.zeros(6144,dtype=torch.float64);absolute=torch.zeros_like(answer)
        for slot,e in enumerate(ids.flatten().tolist()):
            packed=weights[2][e*3072:(e+1)*3072].numpy().reshape(12,256,256)
            scale=weights[3][e*96:(e+1)*96].numpy().reshape(12,8,512)
            original,scale=unpack(packed,scale);codes=np.empty((6144,256),np.uint8);codes[:,::2]=original&15;codes[:,1::2]=original>>4
            dense=q.double()[torch.from_numpy(codes).long()]*torch.pow(2.,torch.from_numpy(scale).double()-127).repeat_interleave(32,dim=1)
            x=gate[slot].double();r=routing[0,slot].double();answer+=(dense@x)*r;absolute+=(dense.abs()@x.abs())*r.abs()
        return answer.reshape(1,-1),absolute.reshape(1,-1)
    # New GUID + unchanged old ELF must pass before testing changed scheduling.
    direct_data=[]
    for mode in (0,1):
        gate=(torch.randn(8,256)*(.125 if mode==0 else 8.)).bfloat16()
        ids=(torch.arange(8,dtype=torch.int32).reshape(1,8)*(1 if mode==0 else 3)+mode)%E
        routing=torch.linspace(.015625,.25,8).reshape(1,8)+2**-18
        if mode:routing*=torch.tensor([[1,-2,4,-8,16,-32,64,-128]],dtype=torch.float32)
        ah,ih,rh=[x.to('hpu') for x in (gate,ids,routing)];sync()
        old=torch.ops.gaudi_down_activation.broadcast(dp,ds,ah,lut,ih);control=ops.old_body(dp,ds,ah,lut,ih);sync()
        oldcpu,controlcpu=old.cpu(),control.cpu();record=dict(mode=mode,old_body_control=summary(oldcpu,controlcpu),variants={})
        result['direct'].append(record);save();assert same(oldcpu,controlcpu) and bool(torch.isfinite(oldcpu).all())
        expected=precision_ops.combine(old,rh,directions,6144,8);sync();expected=expected.cpu()
        oracle,sumabs=down_reference(gate,ids,routing);error=(expected.double()-oracle).abs()
        record['fp64_old_combine']=dict(failures=int((error>2e-6*sumabs+1e-35).sum()),max_abs=float(error.max()),relative_l2=float(error.norm()/oracle.norm().clamp_min(1e-300)),max_sumabs_backward_error=float((error/sumabs.clamp_min(1e-300)).max()))
        assert record['fp64_old_combine']['failures']==0
        saved=dict(mode=mode,gate=gate,ids=ids,routing=routing,old_partial=oldcpu,wrapper_partial=controlcpu,expected=expected,oracle=oracle,sumabs=sumabs)
        for variant in a.variants:
            y=fused(variant,ah,ih,rh);sync();cpu=y.cpu();record['variants'][variant]=summary(expected,cpu);saved[variant]=cpu
            direct_data.append(saved.copy());torch.save(direct_data,out/'direct.pt');save();assert same(expected,cpu) and bool(torch.isfinite(cpu).all())
    # Both arms use the exact same GP, gate and original FP32 route clone.
    xcpu=(torch.randn(1,6144)*.1).bfloat16();xcpu0=xcpu.clone();gaincpu=torch.ones(1,8)
    x=xcpu.to('hpu');gain=gaincpu.to('hpu');offset=torch.tensor(.015625,dtype=torch.bfloat16).to('hpu');post=torch.tensor(.03125,dtype=torch.float32).to('hpu')
    group_count=48 if a.streaming else 1;logits=[];logit_cpus=[]
    for group in range(group_count):
        value=torch.full((1,E),-20.,dtype=torch.float32);value[0,group*8:group*8+8]=torch.arange(8,dtype=torch.float32)/8
        logit_cpus.append(value);logits.append(value.to('hpu'))
    sync();graphs={};outputs={};streams={};variants=['baseline',*a.variants]
    def chain(variant,group):
        produced=(x+offset).clone();values,ids=torch.topk(logits[group]+.125,8,dim=-1)
        ids=ids.to(torch.int32).clone();routing=(torch.softmax(values,-1,dtype=torch.float32)*gain).float().clone()
        partial=torch.ops.gaudi_gp_scale_tail.gp(gp,gs,produced,lut,ids)
        gate=torch.ops.unified_batch.gate(partial,ids)
        if variant=='baseline':
            down=torch.ops.gaudi_down_activation.broadcast(dp,ds,gate,lut,ids)
            value=precision_ops.combine(down,routing,directions,6144,8)
        else:value=fused(variant,gate,ids,routing.float().clone())
        return value+post
    for variant in variants:
        graphs[variant]=[];outputs[variant]=[];streams[variant]=torch.hpu.Stream()
        for group in range(group_count):
            before=graph_ids();stream=streams[variant];graph=torch.hpu.HPUGraph()
            with torch.hpu.stream(stream):
                graph.capture_begin();value=chain(variant,group);graph.capture_end()
            stream.synchronize();sync();graphs[variant].append(graph);outputs[variant].append(value)
            result['records'].append(dict(variant=variant,group=group,postgraph_ids=sorted(graph_ids()-before)))
    from audit_graph import audit
    result['placement']=audit(out/'post_graph.json',E,a.variants);(out/'placement.json').write_text(json.dumps(result['placement'],indent=2)+'\n')
    save();states=['initial','input_hot','cold_signed_routes','weight_owner_changed','zero_routes'] if not a.streaming else ['initial','input_changed']
    owner_ids=[id(t) for t in (gp,gs,dp,ds)];checks=[]
    for state in states:
        if state=='input_hot':x.copy_((xcpu0.roll(17,-1)*1.125).bfloat16())
        elif state=='cold_signed_routes':
            value=torch.full((1,E),-20.,dtype=torch.float32);value[0,-8:]=torch.arange(8,dtype=torch.float32)/8;logits[0].copy_(value);gain.copy_(torch.tensor([[1,-2,4,-8,16,-32,64,-128]],dtype=torch.float32))
        elif state=='weight_owner_changed':
            logits[0].copy_(logit_cpus[0]);gain.copy_(gaincpu);weights[2][:256].bitwise_xor_(1);weights[3][:8].add_(1)
            dp[:256].copy_(weights[2][:256].to('hpu'));ds[:8].copy_(weights[3][:8].to('hpu'));torch.save(dict(dp_block=weights[2][:256].clone(),ds_block=weights[3][:8].clone(),description='first down N512 block XOR1 packed; scales+1'),out/'owner-mutation.pt')
        elif state=='zero_routes':gain.zero_()
        elif state=='input_changed':x.copy_((-xcpu0.roll(31,-1)*.75).bfloat16())
        sync();torch.save(dict(x=x.cpu(),gain=gain.cpu(),logits=[v.cpu() for v in logits]),out/(state+'-inputs.pt'));state_values=[]
        for group in range(group_count):
            actual=[]
            for variant in variants:
                for _ in range(3 if a.streaming else 10):graphs[variant][group].replay(asynchronous=True)
                sync();actual.append(outputs[variant][group].cpu())
            item=dict(state=state,group=group,comparisons={v:summary(actual[0],y) for v,y in zip(a.variants,actual[1:])},same_live_owner_handles=owner_ids==[id(t) for t in (gp,gs,dp,ds)])
            checks.append(item);state_values.append(dict(group=group,values=dict(zip(variants,actual))))
            result['whole_chain_checks']=checks;save();assert all(same(actual[0],y) for y in actual[1:]) and bool(torch.isfinite(actual[0]).all())
        torch.save(state_values,out/(state+'.pt'))
    # All performance inputs are nonzero; reset after the zero-route gate.
    x.copy_(xcpu0);gain.copy_(gaincpu)
    for target,value in zip(logits,logit_cpus):target.copy_(value)
    sync();repeats=48 if a.streaming else 16
    if a.timing:
        for variant in variants:
            for i in range(5):graphs[variant][i%group_count].replay(asynchronous=True)
        sync()
        for candidate in a.variants:
            for trial in range(3):
                for variant in ('baseline',candidate,candidate,'baseline'):
                    stream=streams[variant];start=torch.hpu.Event(enable_timing=True);end=torch.hpu.Event(enable_timing=True)
                    sync();wall=time.perf_counter();start.record(stream)
                    for i in range(repeats):graphs[variant][i%group_count].replay(asynchronous=True)
                    sync();end.record(stream);end.synchronize();sync();elapsed=time.perf_counter()-wall
                    result['timing'].append(dict(candidate=candidate,trial=trial,variant=variant,replays=repeats,expert_groups=group_count,event_us=start.elapsed_time(end)*1000/repeats,wall_us=elapsed*1e6/repeats))
        result['event_wall_consistent']=all(r['event_us']>0 and .8<=r['wall_us']/r['event_us']<=2 for r in result['timing'])
        assert result['event_wall_consistent'],'capture-stream event/wall consistency failed'
        result['medians']={c:{v:{kind:statistics.median(r[kind] for r in result['timing'] if r['candidate']==c and r['variant']==v) for kind in ('event_us','wall_us')} for v in ('baseline',c)} for c in a.variants}
    result.update(status='PASS_BITS_OWNER_DESCRIPTOR_AND_REPLAY',timing_requested=a.timing,all_numeric_pass=True,streaming_expert_count=E if a.streaming else None,streaming_weight_working_set_bytes=result['weights']['bytes'] if a.streaming else None)
except BaseException as error:result.update(status='FAIL',all_numeric_pass=False,error=repr(error));save();raise
finally:save()
print(json.dumps({k:v for k,v in result.items() if k in ('status','medians','weights')},indent=2))
