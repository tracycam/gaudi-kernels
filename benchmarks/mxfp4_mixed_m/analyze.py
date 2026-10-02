"""Audit mixed-M projection results, including rejected expanded-DRAM graphs."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mxfp4_pipeline'))
from isa_identity import text_section
spec = importlib.util.spec_from_file_location('ports', Path(__file__).resolve().parents[1] / 'sram_ports/analyze.py')
ports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ports)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def graph(case, p):
    path = case / 'post_graph.json'
    path = next(path.glob('*.json')) if path.is_dir() else path
    g = json.loads(path.read_text())['graphs'][0]
    ts = {t['name']: t for t in g['tensors']}
    ns = [n for n in g['nodes'] if not n['is_logical']]
    ds = [n for n in ns if n['guid'] in ('gk_grouped_mxfp4_decode', 'gk_mixed_m_decode_prepare')]
    ms = [n for n in ns if n['engine'] == 'MME']
    prep = [n for n in ns if n['guid'].startswith('gk_mixed_m_prepare')]
    assert ds and len(ds) == len(ms) and len(ns) == len(ds) + len(ms) + len(prep)
    for name in ('packed', 'scales'):
        assert ts[name]['dtype'] == 'uint8' and ts[name]['allocation'] == 'DRAM' and ts[name]['persistent']
    base = ts['expert_ids' if p['policy'] in ('fused', 'independent') else 'prepared_ids']
    covered, buffers, names = [], [], set()
    for n in ds:
        assert n['input_tensors'][:2] == ['packed', 'scales']
        idx = ts[n['input_tensors'][3]]
        w = ts[n['output_tensors'][0]]
        start = (idx['offset'] - base['offset']) // 4
        count = idx['max_shape'][0]
        assert w['max_shape'] == [p['N'], p['K'], count] and w['dtype'] == 'bf16' and not w['persistent']
        covered.extend(range(start, start + count))
        names.add(w['name'])
        buffers.append(dict(name=w['name'],allocation=w['allocation'],address=w['offset'],
                            bytes=2 * math.prod(w['max_shape']),expert_start=start,expert_count=count))
    assert sorted(covered) == list(range(p['experts']))
    for n in ms:
        assert len(set(n['input_tensors']) & names) == 1
        assert ts[n['output_tensors'][0]]['dtype'] == 'float32'
    spans = ports.merge([(b['address'], b['address'] + b['bytes']) for b in buffers if b['allocation'] == 'SRAM'])
    capacity = sum(b-a for a,b in spans)
    assert capacity <= 48 * 1024 * 1024
    return dict(graph_sha256=digest(path),all_expanded_weights_sram=all(b['allocation']=='SRAM' for b in buffers),
                expert_coverage_once=True,buffers=buffers,physical_nodes=len(ns),decoder_nodes=len(ds),
                mme_nodes=len(ms),prepare_nodes=len(prep),sram_address_union_bytes=capacity,
                mme_strategies=sorted(set(n.get('mme_node_strategy','') for n in ms)))


def correctness(case, p, runtime):
    e, n, k = (p[x] for x in ('experts', 'N', 'K'))
    order = np.fromfile(case/'slot_to_expert.bin',dtype='<i4')
    offsets = np.fromfile(case/'row_offsets.bin',dtype='<i4')
    assert sorted(order) == list(range(e))
    inverse = np.argsort(order)
    cap = np.diff(np.r_[offsets, p['row_slots']])
    counts = (case/'counts.bin').read_bytes()
    activation = (case/'activation.bin').read_bytes()
    ids = (case/'expert_ids.bin').read_bytes()
    runtime = runtime or dict(changing_counts=False,activation_stride=e*3*k*2,counts_stride=e*4,id_stride=(e*4+255)//256*256)
    records = []
    wh = digest(case/'packed.bin') + digest(case/'scales.bin')
    for b in range(p['banks']):
        ab = b if runtime['changing_counts'] else 0
        co = ab * runtime['counts_stride']
        cp = np.frombuffer(counts[co:co+e*4],dtype='<i4')
        assert np.all((cp >= 1) & (cp <= 3)) and np.all(cp[order] <= cap)
        xo = ab * runtime['activation_stride']
        raw_x = activation[xo:xo+e*3*k*2]
        x = np.frombuffer(raw_x,dtype='<u2').reshape(e,3,k)
        for expert in range(e):
            assert np.all(x[expert,cp[expert]:] == 0x7fc1)
            assert np.all((x[expert,:cp[expert]] & 0x7f80) != 0x7f80)
        io = b * runtime['id_stride']
        ib = ids[io:io+e*4]
        selected = np.frombuffer(ib,dtype='<i4')
        assert len(set(selected)) == e and selected.min() >= 0 and selected.max() < p['pool']
        output = np.fromfile(case/f'output_bank{b}.bin',dtype='<f4').reshape(p['row_slots'],n)
        ref = np.fromfile(case/f'oracle_bank{b}.bin',dtype='<f8').reshape(output.shape)
        assert np.isfinite(output).all()
        rel = float(np.linalg.norm(output-ref)/max(np.linalg.norm(ref),1e-30))
        assert rel < 2e-6
        canonical=[]
        for expert in range(e):
            s = inverse[expert]
            canonical.append(output[offsets[s]:offsets[s]+cp[expert]])
            assert np.all(output[offsets[s]+cp[expert]:offsets[s]+cap[s]] == 0)
        status = case/f'capacity_status_bank{b}.bin'
        if status.exists():assert np.all(np.fromfile(status,dtype='<i4') == 0)
        data = np.concatenate(canonical).tobytes()
        key = hashlib.sha256(wh.encode()+raw_x+cp.tobytes()+ib).hexdigest()
        records.append(dict(bank=b,counts=cp.tolist(),useful_rows=int(cp.sum()),relative_l2=rel,
                            output_sha256=digest(case/f'output_bank{b}.bin'),
                            canonical_output_sha256=hashlib.sha256(data).hexdigest(),input_signature=key))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();records=[];isa=[];prep_isa=[]
    for root in a.roots:
        obj=root/'decoder/decode.o'
        if obj.exists():isa.append(dict(root=str(root),decoder_text_sha256=hashlib.sha256(text_section(obj)).hexdigest()))
        for name in ('prepare','bounded'):
            asm=root/'build'/f'{name}.s';elf=asm.with_suffix('.o')
            if not asm.exists():continue
            source=asm.read_text().split('\t.type\ttpc_compiler')[0]
            local=[l.strip() for l in source.splitlines() if re.search(r'\b(?:ld|st)_l(?:_v)?\b',l) and 'mmio' not in l]
            assert not local
            prep_isa.append(dict(root=str(root),name=name,text_sha256=hashlib.sha256(text_section(elf)).hexdigest(),
                                 max_vector_register=max(int(x) for x in re.findall(r'%V(\d+)',source)),
                                 loops=[l.strip() for l in source.splitlines() if l.strip().startswith('loop ')],
                                 non_mmio_local_instructions=local))
        for case in sorted((root/'results').iterdir()):
            status=json.loads((case/'exit.json').read_text())
            assert status['module']==5 and status['source_identity_verified'] and status['runner_exit_code']==0
            rows=[json.loads(s) for s in (case/'run.log').read_text().splitlines() if s.startswith('{"stage":')]
            p=next(x for x in rows if x['stage']=='plan')
            r=dict(root=str(root),case=case.name,source_commit=status['git_commit'],plan=p,placement=graph(case,p))
            compile_only=any(x['stage']=='compile_only' for x in rows)
            r['compile_only']=compile_only
            if not r['placement']['all_expanded_weights_sram']:
                assert compile_only
                r.update(accepted=False,rejection='Expanded weights in DRAM; compile-only, no replay.')
            elif not compile_only:
                check=next(x for x in rows if x['stage']=='correctness');assert check['bad']==0
                runtime=next((x for x in rows if x['stage']=='runtime_inputs'),None)
                samples=[x['event_us'] for x in rows if x['stage']=='timing'];assert len(samples)==5
                us=statistics.median(samples);weight_elements=p['N']*p['K']*p['experts']
                r.update(accepted=True,correctness=check,runtime_inputs=runtime,samples_us=samples,event_median_us=us,
                         banks=correctness(case,p,runtime),payload_TBps=weight_elements*.5/us/1e6,
                         original_TBps=weight_elements*17/32/us/1e6,expanded_feed_TBps=weight_elements*2/us/1e6)
                traces=list((case/'trace').glob('*_7.json'))
                if traces:
                    tr=ports.trace(traces[0],None);r['profile']=tr
                    prep=[n for n in tr['nodes'] if n['name']=='mixed_prepare']
                    decode=sorted((n for n in tr['nodes'] if n['engine']=='TPC' and 'decode' in n['name']),key=lambda n:n['start_us'])
                    assert len(prep)==1 and decode
                    tr['prep_duration_us']=prep[0]['end_us']-prep[0]['start_us']
                    tr['first_decode_start_us']=decode[0]['start_us']
                    tr['prep_decode_envelope_overlap_us']=max(0,min(prep[0]['end_us'],decode[0]['end_us'])-max(prep[0]['start_us'],decode[0]['start_us']))
            records.append(r)
    assert len({i['decoder_text_sha256'] for i in isa})==1
    # Prep changes must leave identical MME arithmetic and every output bit intact.
    keys={}
    for r in records:
        if not r.get('accepted'):continue
        if r['plan']['policy'] not in ('padded','bounded','independent'):continue
        for b in r['banks']:
            key=b['input_signature'];sha=b['canonical_output_sha256']
            assert key not in keys or keys[key]==sha
            keys[key]=sha
        r['same_arithmetic_bitwise_gate']=True
    abba=[]
    for root in a.roots:
        by={r['case']:r for r in records if r['root']==str(root)}
        for prefix in ('abba','grouping','independence'):
            arms=sorted((r for name,r in by.items() if name.startswith(prefix+'-')),key=lambda r:r['case'])
            if not arms:continue
            assert len(arms)==4 and all(r.get('accepted') and 'profile' not in r for r in arms)
            assert arms[0]['plan']['policy']==arms[3]['plan']['policy'] and arms[1]['plan']['policy']==arms[2]['plan']['policy']
            for i in (1,2,3):assert [b['input_signature'] for b in arms[i]['banks']]==[b['input_signature'] for b in arms[0]['banks']]
            before=statistics.mean(arms[i]['event_median_us'] for i in (0,3))
            after=statistics.mean(arms[i]['event_median_us'] for i in (1,2))
            p=arms[1]['plan'];elements=p['experts']*p['N']*p['K']
            abba.append(dict(root=str(root),prefix=prefix,arms=[r['case'] for r in arms],baseline_us=before,candidate_us=after,
                             baseline_policy=arms[0]['plan']['policy'],candidate_policy=arms[1]['plan']['policy'],
                             latency_reduction=1-after/before,payload_TBps=elements*.5/after/1e6,
                             original_TBps=elements*17/32/after/1e6,expanded_feed_TBps=elements*2/after/1e6))
    result=dict(scope='One device graph per already-routed mixed-M projection; includes preparation. Not complete MoE or model TPS.',
                decoder_isa=isa,preparation_isa=prep_isa,ABBA=abba,records=records,
                caveats=['Logical weight coverage and SRAM placement audited; physical bus transactions not counted.',
                         'Shape-specific pair/bucket capacities cannot accept arbitrary changing counts.',
                         'Profile markers are not SRAM read-start events. Bandwidth uses complete unprofiled event time.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(abba,indent=2))


if __name__=='__main__':main()
