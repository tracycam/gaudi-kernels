"""Audit actual owners, ELF/ISA, outputs and all samples, including regressions."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import numpy as np

p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tools'))
from moe_activation_fold_core import embedded_elf
sys.path.insert(0,str(repo/'benchmarks/mxfp4_pipeline'))
from isa_identity import text_section
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];builds=[]
for root in a.roots:
    for obj in sorted((root/'builds').glob('*/grouped.o')):
        dis=obj.with_suffix('.o.dis').read_text()
        lines=[s.split('//')[0].strip() for s in re.findall(r'^\s*[0-9a-f]+:\s+(.*)',dis,re.M)]
        local=[s for s in lines if re.search(r'\b(?:ld|st)_l(?:_v)?\b',s) and 'mmio' not in s]
        slots=[s.split(';')for s in lines if ';'in s]
        assert all(len(s)==4 for s in slots)
        assert embedded_elf((obj.parent/'libgrouped_tpc.so').read_bytes(),'grouped')==obj.read_bytes()
        regs=[int(v)+(kind=='D') for kind,v in re.findall(r'\b(V|D)(\d+)\b',dis)]
        builds.append(dict(root=str(root),build=obj.parent.name,elf_sha256=sha(obj),text_bytes=len(text_section(obj)),
                           packets=len(lines),max_vector_register=max(regs),static_local_memory_ops=len(local),
                           all_nop_packets=sum(all(v.strip()=='nop' for v in s)for s in slots),
                           static_slot_ops=[dict(Counter(s[i].strip().split()[0]for s in slots))for i in range(4)],
                           metadata=json.loads((obj.parent/'build.json').read_text()),
                           scope='Static sites, not dynamic cycles or measured issue utilization'))
    for case in sorted((root/'results').iterdir()):
        if not (case/'exit.json').exists() or not (case/'run.log').exists():continue
        exit_=json.loads((case/'exit.json').read_text())
        rows=[json.loads(s)for s in (case/'run.log').read_text().splitlines()if s.startswith('{"stage":')]
        if not rows:continue
        plan=next(r for r in rows if r['stage']=='plan')
        gate=next((r for r in rows if r['stage']=='correctness'),None)
        samples=[r['event_us']for r in rows if r['stage']=='timing']
        record=dict(root=str(root),case=case.name,plan=plan,gate=gate,exit_code=exit_['runner_exit_code'],samples_us=samples)
        records.append(record)
        if exit_['runner_exit_code']!=0 or not gate or gate['bad']:continue
        assert exit_['source_identity_verified'] and exit_['module']==5
        E,P,N,K,M,S,B=[plan[k]for k in ('experts','pool','N','K','M_cap','splits','banks')]
        tile=plan['weight_tile'];assert tile in(128,256)
        paths=list((case/'post_graph.json').rglob('*.json')) if (case/'post_graph.json').is_dir() else [case/'post_graph.json']
        graphs=[g for f in paths for g in json.loads(f.read_text())['graphs']];assert len(graphs)==1
        graph=graphs[0];ts={t['name']:t for t in graph['tensors']}
        shapes={'packed':[128,K//2,N//128,P]if tile==128 else[128,K,N//256,P], 'scales':[tile,K//32,N//tile,P]}
        for name,shape in shapes.items():
            t=ts[name];assert t['max_shape']==shape and t['persistent'] and t['dtype']=='uint8' and t['allocation']=='DRAM'
            assert (case/(name+'.bin')).stat().st_size==math.prod(shape)
        assert plan['rotating_weight_bytes']==P*N*K*17//32 and plan['weight_bytes_per_replay']==E*N*K*17//32
        assert gate['checked']==E*N*M*B
        physical=[n for n in graph['nodes']if not n.get('is_logical',False)]
        assert len(physical)==1+(S>1)
        assert all(n['guid']in('gk_grouped_mxfp4_smallm','gk_grouped_mxfp4_reduce')for n in physical)
        assert all(t['dtype']!='bf16' or t['name'].startswith(('activation','byte_lut'))for t in ts.values())
        ids=np.fromfile(case/'expert_ids.bin',dtype='<i4').reshape(B,-1)[:,:E]
        assert all(len(set(row))==E for row in ids) and sorted(ids.ravel())==list(range(P))
        packed=np.memmap(case/'packed.bin',dtype=np.uint8,mode='r',shape=(P,N*K//2))
        scales=np.memmap(case/'scales.bin',dtype=np.uint8,mode='r',shape=(P,N*K//32))
        digest=hashlib.sha256()
        for e in range(P):
            if tile==128:
                q=packed[e].reshape(N//128,K//2,128)
                q=np.stack((q&15,q>>4),axis=2).reshape(N//128,K,128)
            else:
                q=packed[e].reshape(N//256,K,128);q=np.concatenate((q&15,q>>4),axis=2)
            digest.update(q.transpose(0,2,1).copy().tobytes())
            digest.update(scales[e].reshape(N//tile,K//32,tile).transpose(0,2,1).copy().tobytes())
        out_hashes={f.name:sha(f)for pattern in ('output_bank*.bin','oracle_bank*.bin','activation.bin','counts.bin','expert_ids.bin')for f in case.glob(pattern)}
        us=statistics.median(samples)
        record.update(graph_pass=True,semantic_original_owner_sha256=digest.hexdigest(),comparand_sha256=out_hashes,
                      event_median_us=us,effective_original_TBps=E*N*K*17/32/us/1e6,
                      useful_TFLOPs=2*plan['valid_rows']*N*K/us/1e6,
                      libraries=exit_['kernel_libraries_sha256'])
groups={}
for r in records:
    if not r.get('graph_pass'):continue
    key=tuple(r['plan'][k]for k in('experts','pool','N','K','M_cap','splits','banks','pattern'))
    groups.setdefault(key,[]).append(r)
for key,rs in groups.items():
    assert len({r['semantic_original_owner_sha256']for r in rs})==1,(key,'logical weights/scales changed')
    assert all(r['comparand_sha256']==rs[0]['comparand_sha256']for r in rs),(key,'inputs or outputs changed')
abba=[]
for root in a.roots:
    selected={r['case']:r for r in records if r['root']==str(root)}
    for kind in ('gp','down'):
        names=[f'abba-{kind}-{i}'for i in range(4)]
        if not all(n in selected for n in names):continue
        arms=[selected[n]for n in names]
        assert all(r.get('graph_pass')and r['exit_code']==0 for r in arms)
        assert all(r['comparand_sha256']==arms[0]['comparand_sha256']for r in arms)
        old=statistics.mean(arms[0]['samples_us']+arms[3]['samples_us'])
        new=statistics.mean(arms[1]['samples_us']+arms[2]['samples_us'])
        plan=arms[0]['plan']
        abba.append(dict(root=str(root),projection=kind,arms=names,control_us=old,candidate_us=new,
                         latency_reduction=1-new/old,original_TBps=plan['weight_bytes_per_replay']/new/1e6,
                         useful_TFLOPs=2*plan['valid_rows']*plan['N']*plan['K']/new/1e6,
                         all_inputs_outputs_bitwise_equal=True,plan=plan))
result=dict(records=records,builds=builds,ABBA=abba,matching_geometries_bitwise_equal=True,
            scope='Synthetic grouped projection including split reduction. Original byte ownership verified; physical HBM transactions and full-model gain not inferred.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in records:print(json.dumps({k:r[k]for k in('case','exit_code','event_median_us','effective_original_TBps','useful_TFLOPs')if k in r}))
