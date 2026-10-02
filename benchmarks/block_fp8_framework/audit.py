"""Audit physical PostGraph placement and logical prepared-weight coverage.

Rectangular view ranges use actual strides/offsets, not names or allocation
flags alone. Summed tensor write volumes are not physical HBM transactions.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def ranges(t, root):
    shape=t['max_shape'];strides=t['strides'];item=t['dtype_bit_size']//8
    span=item;axis=0
    while axis<len(shape) and strides[axis]==span:
        span*=shape[axis];axis+=1
    base=t['offset']-root['offset']
    for index in itertools.product(*(range(n) for n in shape[axis:])):
        start=base+sum(i*s for i,s in zip(index,strides[axis:]))
        yield (start,start+span)


def audit(graph):
    tensors={t['name']:t for t in graph['tensors']}
    nodes=[n for n in graph['nodes'] if not n['is_logical']]
    def original(t):
        seen=set()
        while t.get('alias') and t.get('alias_of') in tensors:
            assert t['name'] not in seen;seen.add(t['name']);t=tensors[t['alias_of']]
        return t
    reads={};writes=[];decoded=[]
    for node in nodes:
        candidates=[]
        if node['guid'].startswith('gk_block128_decode'):candidates=node['input_tensors'][:1]
        elif node['engine']=='MME':
            # These registered GEMMs use input0=A,input1=W (transpose_b=true).
            # External quantized activations can also be persistent; they must
            # not be mistaken for a second prepared weight owner.
            candidates=[n for n in node['input_tensors'][1:2] if tensors[n]['dtype_bit_size']==8
                        and original(tensors[n])['persistent']]
        for name in candidates:
            t=tensors[name];r=original(t)
            assert r['persistent'] and r['dtype_bit_size']==8
            reads.setdefault(r['name'],{'root':r,'ranges':[]})['ranges'].extend(ranges(t,r))
        for name in node['output_tensors']:
            t=tensors[name]
            row={'node':node['guid'],'tensor':name,'allocation':t['allocation'],
                 'bytes':math.prod(t['max_shape'])*t['dtype_bit_size']//8,
                 'persistent_storage':original(t)['persistent']}
            writes.append(row)
            if node['guid'].startswith('gk_block128_decode'):decoded.append(row)
    weight=[]
    for value in reads.values():
        root=value['root'];intervals=sorted(value['ranges']);end=0;union=0;total=0
        for a,b in intervals:
            assert 0<=a<=b<=math.prod(root['max_shape'])
            total+=b-a
            if b>end:union+=b-max(a,end);end=b
        expected=math.prod(root['max_shape'])
        weight.append({'tensor':root['name'],'prepared_bytes':expected,'logical_read_bytes':total,
                       'covered_bytes':union,'coverage_complete':union==expected,
                       'logical_read_ratio':total/expected})
    transient=[t for t in writes if t['allocation']=='DRAM' and not t['persistent_storage']]
    assert weight and all(w['coverage_complete'] for w in weight)
    assert all(t['allocation']=='SRAM' and not t['persistent_storage'] for t in decoded),'expanded decoded weight is not transient SRAM'
    return {'graph':graph['name'],'workspace_bytes':graph['workspace_size'],
            'physical_compute_nodes':sum(n['engine'] in ('TPC','MME') for n in nodes),
            'physical_dma_nodes':sum(n['engine']=='DMA' for n in nodes),
            'weight_reads':weight,'decoded_weight_outputs':decoded,
            'transient_dram_output_write_volume':sum(t['bytes'] for t in transient),
            'transient_dram_outputs':transient,
            'no_logical_weight_read_amplification':all(w['logical_read_ratio']==1 for w in weight)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=json.loads((a.directory/'result.json').read_text());graphs=json.loads((a.directory/'post_graph.json').read_text())['graphs']
    # Standalone quant byte-gate graphs are diagnostic outputs, not Linear.
    graphs=[g for g in graphs if any(n['engine']=='MME' for n in g['nodes'])]
    assert len(graphs)==len(result['records'])
    records=[]
    for row,graph in zip(result['records'],graphs):
        placement=audit(graph);case=row['case'];m,n,k=case['shape'];d=case.get('depth',1)
        payload=n*k+(d-1)*n*n;us=row['median_event_us']
        timing_valid=max(us,row['median_wall_us'])/min(us,row['median_wall_us'])<1.25
        records.append({'case':case,'event_us':us,'wall_us':row['median_wall_us'],
            'timing_valid':timing_valid,'original_weight_payload_bytes':payload,
            'effective_TFLOPS':2*m*payload/(us*1e6) if timing_valid else None,
            'payload_TBps':payload/(us*1e6) if timing_valid else None,'placement':placement})
    output={'status':'PASS_AUDIT' if all(r['placement']['no_logical_weight_read_amplification'] for r in records) else 'REJECTED_WEIGHT_READ_AMPLIFICATION',
            'probe_status':result['status'],'scope':'PostGraph logical byte coverage/placement, not BMON traffic or hardware launch counts','records':records}
    if result['status']!='PASS_NUMERICAL_PLACEMENT_UNAUDITED':output['status']='PARTIAL_RUN_AUDITED_RECORDS_ONLY'
    elif not all(r['timing_valid'] for r in records):output['status']='PASS_PLACEMENT_TIMING_INVALID'
    (a.directory/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'records':[{k:r[k] for k in ['case','event_us','effective_TFLOPS']}|{'dram_write_volume':r['placement']['transient_dram_output_write_volume'],'weight_read_ratios':[w['logical_read_ratio'] for w in r['placement']['weight_reads']]} for r in records]}))
